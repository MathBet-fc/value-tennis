import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 1. CONFIGURAZIONE GLOBALE ---
st.set_page_config(
    page_title="Tennis Quant Pro - Professional Edition",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.style.use('dark_background')
sns.set_style("darkgrid")

# --- 2. CLASSE MATH & UTILS ---
class TennisMath:
    @staticmethod
    def kelly_criterion(prob, odds, fractional=0.25):
        if prob <= 0 or odds <= 1: return 0.0
        b = odds - 1
        q = 1 - prob
        f = ((b * prob) - q) / b
        return max(0.0, f * fractional)

    @staticmethod
    def adjust_stats_for_cpi(serve_pct, ace_pct, cpi):
        factor = (cpi - 35) / 100 
        adj_serve = serve_pct + (factor * 0.8)
        adj_ace = ace_pct * (1 + factor * 1.5)
        return min(0.98, max(0.30, adj_serve)), max(0.0, adj_ace)

    @staticmethod
    def apply_tactical_adjustments(p_srv_win, p_ace, server_hand, returner_bh, conditions):
        adj_srv = p_srv_win
        adj_ace = p_ace
        if server_hand == "Sinistra" and returner_bh == "Una Mano":
            adj_srv *= 1.04; adj_ace *= 1.05
        elif server_hand == "Destra" and returner_bh == "Una Mano":
            adj_srv *= 1.015
        if conditions['altitude']: adj_srv *= 1.02; adj_ace *= 1.10
        if conditions['indoor']: adj_ace *= 1.05; adj_srv *= 1.01
        if conditions['windy']: adj_srv *= 0.96; adj_ace *= 0.90
        return min(0.99, adj_srv), adj_ace

    @staticmethod
    def log5_matchup(srv_win_pct, ret_win_pct, tour_avg_srv):
        tour_avg_ret = 1 - tour_avg_srv
        diff_ret = ret_win_pct - tour_avg_ret
        adj_prob = srv_win_pct - diff_ret
        return min(0.99, max(0.01, adj_prob))

    @staticmethod
    def generate_market_table(data_series, market_name, step=1):
        mean_val = data_series.mean()
        start = max(0.5, np.floor(mean_val) - 5 + 0.5)
        lines = [start + i*step for i in range(11)]
        results = []
        total = len(data_series)
        for line in lines:
            over = (data_series > line).sum()
            under = (data_series < line).sum()
            p_over = over / total
            p_under = under / total
            o_odd = 1/p_over if p_over > 0.01 else 999.0
            u_odd = 1/p_under if p_under > 0.01 else 999.0
            results.append({
                "Linea": line,
                "Over %": f"{p_over:.1%}",
                "Over Quota": f"{o_odd:.2f}",
                "Under %": f"{p_under:.1%}",
                "Under Quota": f"{u_odd:.2f}"
            })
        return pd.DataFrame(results)

# --- 3. CLASSE MARKOV ---
class TennisMarkov:
    @staticmethod
    def prob_hold_game(p, s_srv=0, s_ret=0):
        if s_srv >= 4 and s_srv >= s_ret + 2: return 1.0
        if s_ret >= 4 and s_ret >= s_srv + 2: return 0.0
        if s_srv >= 3 and s_ret >= 3:
            if s_srv == s_ret: return p**2 / (p**2 + (1-p)**2)
            elif s_srv > s_ret: return p + (1-p) * (p**2 / (p**2 + (1-p)**2))
            else: return p * (p**2 / (p**2 + (1-p)**2))
        memo = {}
        def solve(i, j):
            if (i, j) in memo: return memo[(i, j)]
            if i >= 4 and i >= j + 2: return 1.0
            if j >= 4 and j >= i + 2: return 0.0
            if i==3 and j==3: return p**2 / (p**2 + (1-p)**2)
            res = p * solve(i+1, j) + (1-p) * solve(i, j+1)
            memo[(i, j)] = res
            return res
        return solve(s_srv, s_ret)

    @staticmethod
    def get_full_theoretical_prob(p1_stats, p2_stats, best_of=3):
        p1_srv = p1_stats['1st_win']*p1_stats['1st_in'] + p1_stats['2nd_win']*(1-p1_stats['1st_in'])
        p2_srv = p2_stats['1st_win']*p2_stats['1st_in'] + p2_stats['2nd_win']*(1-p2_stats['1st_in'])
        p_hold_p1 = TennisMarkov.prob_hold_game(p1_srv)
        p_hold_p2 = TennisMarkov.prob_hold_game(p2_srv)
        p_avg = (p1_srv + (1 - p2_srv)) / 2
        p_tb = 1 / (1 + np.exp(-12 * (p_avg - 0.5)))
        n_iter = 1000
        p1_sets = 0
        for _ in range(n_iter):
            g1, g2 = 0, 0
            while True:
                if np.random.random() < p_hold_p1: g1+=1
                else: g2+=1
                if (g1>=6 and g1-g2>=2): p1_sets+=1; break
                if (g2>=6 and g2-g1>=2): break
                if np.random.random() < p_hold_p2: g2+=1
                else: g1+=1
                if (g1>=6 and g1-g2>=2): p1_sets+=1; break
                if (g2>=6 and g2-g1>=2): break
                if g1==6 and g2==6:
                    if np.random.random() < p_tb: p1_sets+=1
                    break
        p_set = p1_sets / n_iter
        p = p_set
        if best_of == 3: return p**2 + 2*(p**2)*(1-p), p_set
        else: return p**3 + 3*(p**3)*(1-p) + 6*(p**3)*((1-p)**2), p_set

# --- 4. ML ENGINE ---
class TennisMLPredictor:
    def __init__(self):
        self.data = None

    @st.cache_data(ttl=3600*12) 
    def load_and_prep_data(_self, repo_name):
        try:
            prefix = repo_name.replace('tennis_', '')
            base_url = f"https://raw.githubusercontent.com/JeffSackmann/{repo_name}/master"
            urls = [
                f"{base_url}/{prefix}_matches_2023.csv",
                f"{base_url}/{prefix}_matches_2024.csv",
                f"{base_url}/{prefix}_matches_2025.csv"
            ]
            dfs = []
            for url in urls:
                try: dfs.append(pd.read_csv(url))
                except: pass
            if not dfs: return None
            df = pd.concat(dfs)
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
            return df
        except: return None

    def calculate_tour_baselines(self, df, surface_filter=None):
        if df is None or df.empty: return None, None, None
        data = df[df['surface'] == surface_filter] if surface_filter else df
        if len(data) < 50: data = df 
        tot_1st_in = data['w_1stIn'].sum() + data['l_1stIn'].sum()
        tot_1st_won = data['w_1stWon'].sum() + data['l_1stWon'].sum()
        avg_1st_win = tot_1st_won / tot_1st_in if tot_1st_in > 0 else 0.70
        tot_sv_pts = data['w_svpt'].sum() + data['l_svpt'].sum()
        tot_2nd_played = tot_sv_pts - tot_1st_in
        tot_2nd_won = data['w_2ndWon'].sum() + data['l_2ndWon'].sum()
        avg_2nd_win = tot_2nd_won / tot_2nd_played if tot_2nd_played > 0 else 0.50
        tot_srv_won = tot_1st_won + tot_2nd_won
        avg_total_srv = tot_srv_won / tot_sv_pts if tot_sv_pts > 0 else 0.60
        return avg_1st_win, avg_2nd_win, avg_total_srv

    def get_player_prediction(self, df, player_name, surface):
        if df is None: return None
        p_wins = df[df['winner_name'] == player_name].copy()
        p_wins['opp_1st_in'] = p_wins['l_1stIn']
        p_wins['opp_1st_won'] = p_wins['l_1stWon']
        p_wins['opp_2nd_in'] = p_wins['l_svpt'] - p_wins['l_1stIn']
        p_wins['opp_2nd_won'] = p_wins['l_2ndWon']
        p_wins.rename(columns={'w_1stIn':'1in', 'w_svpt':'svpt', 'w_1stWon':'1w', 'w_2ndWon':'2w', 'w_ace':'ace', 'w_df':'df'}, inplace=True)
        p_loss = df[df['loser_name'] == player_name].copy()
        p_loss['opp_1st_in'] = p_loss['w_1stIn']
        p_loss['opp_1st_won'] = p_loss['w_1stWon']
        p_loss['opp_2nd_in'] = p_loss['w_svpt'] - p_loss['w_1stIn']
        p_loss['opp_2nd_won'] = p_loss['w_2ndWon']
        p_loss.rename(columns={'l_1stIn':'1in', 'l_svpt':'svpt', 'l_1stWon':'1w', 'l_2ndWon':'2w', 'l_ace':'ace', 'l_df':'df'}, inplace=True)
        if p_wins.empty and p_loss.empty: return None
        hist = pd.concat([p_wins, p_loss], ignore_index=True)
        hist = hist.sort_values('tourney_date', ascending=False).head(50) 
        surf_hist = hist[hist['surface'] == surface]
        if len(surf_hist) >= 5: hist = surf_hist
        tot_svpt = hist['svpt'].sum()
        if tot_svpt == 0: return None
        tot_opp_1st_in = hist['opp_1st_in'].sum()
        tot_opp_1st_won = hist['opp_1st_won'].sum()
        ret_1st = 1.0 - (tot_opp_1st_won / tot_opp_1st_in) if tot_opp_1st_in > 0 else 0.30
        tot_opp_2nd_in = hist['opp_2nd_in'].sum()
        tot_opp_2nd_won = hist['opp_2nd_won'].sum()
        ret_2nd = 1.0 - (tot_opp_2nd_won / tot_opp_2nd_in) if tot_opp_2nd_in > 0 else 0.50
        return {
            '1st_in': hist['1in'].sum() / tot_svpt,
            '1st_win': hist['1w'].sum() / hist['1in'].sum() if hist['1in'].sum()>0 else 0,
            '2nd_win': hist['2w'].sum() / (tot_svpt - hist['1in'].sum()) if (tot_svpt - hist['1in'].sum())>0 else 0,
            'ace_pct': hist['ace'].sum() / tot_svpt,
            'df_pct': hist['df'].sum() / tot_svpt,
            'ret_1st_win': ret_1st, 'ret_2nd_win': ret_2nd
        }

# --- 5. MONTE CARLO ENGINE ---
class TennisMonteCarloEngine:
    def __init__(self, p1_stats, p2_stats, config, current_state=None):
        self.orig_p1 = p1_stats
        self.orig_p2 = p2_stats
        self.n = config['simulations']
        self.match_sets = config.get('sets_to_win', 2)
        self.state = current_state
        self.p1 = None
        self.p2 = None

    @staticmethod
    def simulate_current_game_mc(p_base, srv_score, ret_score, mental_factor, n=1000):
        wins = 0
        for _ in range(n):
            s, r = srv_score, ret_score
            while True:
                if s >= 4 and s >= r + 2: wins += 1; break
                if r >= 4 and r >= s + 2: break
                if s == 4 and r == 4: s = 3; r = 3 
                is_bp = (r == 3 and s < 3) or (r == 4 and s == 3)
                current_p = p_base
                if is_bp: current_p = p_base * (1.0 + (mental_factor - 1.0) * 1.5)
                if np.random.random() < current_p: s += 1
                else: r += 1
        return wins / n

    def _play_point(self, server_id, pressure_level=1.0):
        s = self.p1 if server_id == 1 else self.p2
        perf_boost = 1.0 + (s['mental'] - 1.0) * pressure_level
        if np.random.random() < s['1st_in']:
            if np.random.random() < s['ace_pct'] * perf_boost: return 'ace'
            win_prob = min(0.99, s['1st_win'] * perf_boost)
            return 'srv' if np.random.random() < win_prob else 'ret'
        else:
            real_df_prob = s['df_pct'] / perf_boost 
            if np.random.random() < real_df_prob: return 'df'
            win_prob = min(0.99, s['2nd_win'] * perf_boost)
            return 'srv' if np.random.random() < win_prob else 'ret'

    def _simulate_game_detailed(self, server_id, p_srv_start=0, p_ret_start=0):
        pts_srv = p_srv_start; pts_ret = p_ret_start
        stats = {'aces': 0, 'dfs': 0, 'bps_faced': 0}
        while True:
            is_bp = (pts_ret == 3 and pts_srv < 3) or (pts_ret == 4 and pts_srv == 3)
            if is_bp: stats['bps_faced'] += 1
            pressure = 1.5 if is_bp else 0.5
            res = self._play_point(server_id, pressure)
            if res in ['ace', 'srv']:
                pts_srv += 1; 
                if res == 'ace': stats['aces'] += 1
            else:
                pts_ret += 1; 
                if res == 'df': stats['dfs'] += 1
            if pts_srv >= 4 and pts_srv >= pts_ret + 2: return server_id, stats
            if pts_ret >= 4 and pts_ret >= pts_srv + 2: return (2 if server_id==1 else 1), stats
            if pts_srv == 4 and pts_ret == 4: pts_srv = 3; pts_ret = 3

    def _simulate_tie_break(self, start_p1=0, start_p2=0):
        p1, p2 = start_p1, start_p2
        stats = {'aces_p1':0, 'aces_p2':0, 'dfs_p1':0, 'dfs_p2':0}
        while True:
            server = 1 if (p1 + p2) % 4 in [0, 3] else 2
            res = self._play_point(server, pressure_level=1.2)
            if res in ['ace', 'srv']:
                if server == 1: p1 += 1; stats['aces_p1'] += (1 if res=='ace' else 0)
                else: p2 += 1; stats['aces_p2'] += (1 if res=='ace' else 0)
            else:
                if server == 1: p2 += 1; stats['dfs_p1'] += (1 if res=='df' else 0)
                else: p1 += 1; stats['dfs_p2'] += (1 if res=='df' else 0)
            if (p1 >= 7 and p1 >= p2 + 2): return 1, stats
            if (p2 >= 7 and p2 >= p1 + 2): return 2, stats

    def run(self):
        results = []
        p1_wins = 0; count_tb = 0; count_comeback = 0; set_scores = {}
        s_sets1 = self.state['sets_p1'] if self.state else 0
        s_sets2 = self.state['sets_p2'] if self.state else 0
        s_gms1 = self.state['games_p1'] if self.state else 0
        s_gms2 = self.state['games_p2'] if self.state else 0
        s_pts1 = self.state['pts_p1'] if self.state else 0
        s_pts2 = self.state['pts_p2'] if self.state else 0
        s_server = self.state['server'] if self.state else 1
        
        for _ in range(self.n):
            self.p1 = self.orig_p1.copy()
            self.p2 = self.orig_p2.copy()
            sets_p1, sets_p2 = s_sets1, s_sets2
            g1, g2 = s_gms1, s_gms2
            curr_server = s_server
            match_stats = {'p1_aces':0, 'p2_aces':0, 'p1_breaks':0, 'p2_breaks':0}
            total_games = g1 + g2
            score_log = [] 
            has_tb = False
            
            if self.state and (s_pts1 > 0 or s_pts2 > 0):
                w, s = self._simulate_game_detailed(curr_server, s_pts1, s_pts2)
                if curr_server==1: match_stats['p1_aces']+=s['aces']
                else: match_stats['p2_aces']+=s['aces']
                if w==1: 
                    g1+=1; 
                    if curr_server==2: match_stats['p1_breaks']+=1
                else: 
                    g2+=1; 
                    if curr_server==1: match_stats['p2_breaks']+=1
                curr_server = 2 if curr_server == 1 else 1

            while sets_p1 < self.match_sets and sets_p2 < self.match_sets:
                if len(score_log) > 0:
                    last = score_log[-1]
                    sl1, sl2 = map(int, last.split('-'))
                    if abs(sl1 - sl2) >= 4:
                        if sl1 > sl2: self.p2['mental'] *= 0.95
                        else: self.p1['mental'] *= 0.95
                if sets_p1 == self.match_sets - 1 and sets_p2 == self.match_sets - 1:
                    self.p1['1st_win'] *= self.p1['fatigue_factor']
                    self.p2['1st_win'] *= self.p2['fatigue_factor']

                while True:
                    if (g1>=6 and g1-g2>=2) or (g1==7 and g2==6):
                        score_log.append(f"{g1}-{g2}"); sets_p1+=1; g1,g2=0,0; break
                    if (g2>=6 and g2-g1>=2) or (g2==7 and g1==6):
                        score_log.append(f"{g1}-{g2}"); sets_p2+=1; g1,g2=0,0; break
                    if g1==6 and g2==6:
                        has_tb=True; tb_w, tb_s = self._simulate_tie_break()
                        match_stats['p1_aces']+=tb_s['aces_p1']; match_stats['p2_aces']+=tb_s['aces_p2']
                        if tb_w==1: score_log.append("7-6"); sets_p1+=1
                        else: score_log.append("6-7"); sets_p2+=1
                        g1,g2=0,0; break
                    
                    w, s = self._simulate_game_detailed(curr_server)
                    if curr_server==1: match_stats['p1_aces']+=s['aces']
                    else: match_stats['p2_aces']+=s['aces']
                    if w==1: 
                        g1+=1; 
                        if curr_server==2: match_stats['p1_breaks']+=1
                    else: 
                        g2+=1; 
                        if curr_server==1: match_stats['p2_breaks']+=1
                    curr_server = 2 if curr_server == 1 else 1
                    total_games += 1
            
            winner = 1 if sets_p1 > sets_p2 else 2
            if winner==1: p1_wins+=1
            if has_tb: count_tb+=1
            if len(score_log)>0:
                s1f, s2f = map(int, score_log[0].split('-'))
                if winner==1 and s1f<s2f: count_comeback+=1
            
            score_key = f"{sets_p1}-{sets_p2}"
            set_scores[score_key] = set_scores.get(score_key, 0) + 1
            results.append({
                'winner': winner, 'tot_games': total_games,
                'tot_sets': sets_p1 + sets_p2,
                'p1_aces': match_stats['p1_aces'], 'p2_aces': match_stats['p2_aces'],
                'tot_aces': match_stats['p1_aces'] + match_stats['p2_aces'],
                'p1_breaks': match_stats['p1_breaks'], 'p2_breaks': match_stats['p2_breaks'],
                'tot_breaks': match_stats['p1_breaks'] + match_stats['p2_breaks'],
                'first_set_score': score_log[0] if score_log else "0-0",
                'diff_games': sum([int(x.split('-')[0]) for x in score_log]) - sum([int(x.split('-')[1]) for x in score_log])
            })
        
        df = pd.DataFrame(results)
        return {
            'win_prob': p1_wins / self.n, 'tb_prob': count_tb / self.n,
            'comeback_prob': count_comeback / self.n, 'set_betting': {k: v/self.n for k,v in set_scores.items()},
            'avg_aces_p1': df['p1_aces'].mean(), 'avg_aces_p2': df['p2_aces'].mean(),
            'avg_breaks_p1': df['p1_breaks'].mean(), 'avg_breaks_p2': df['p2_breaks'].mean()
        }, df

# --- 6. MAIN INTERFACE ---
def main():
    # --- INIT VARIABLES ---
    p1_name, p2_name = "Giocatore 1", "Giocatore 2"
    p1_1w, p2_1w = 0.65, 0.65
    p1_1in, p2_1in = 0.60, 0.60
    p1_2w, p2_2w = 0.50, 0.50
    p1_ace, p2_ace = 0.05, 0.05
    p1_df, p2_df = 0.03, 0.03
    p1_r1, p2_r1 = 0.30, 0.30
    p1_r2, p2_r2 = 0.50, 0.50
    
    st.sidebar.title("🛠️ Setup Match")
    
    circuit = st.sidebar.radio("Circuito", ["ATP (Uomini)", "WTA (Donne)"])
    repo_name = "tennis_atp" if circuit == "ATP (Uomini)" else "tennis_wta"
    sets_to_win = 2
    if circuit == "ATP (Uomini)" and st.sidebar.checkbox("Slam Mode (Best of 5)"): sets_to_win = 3

    ml = TennisMLPredictor()
    
    if 'curr_repo' not in st.session_state: st.session_state.curr_repo = repo_name
    if st.session_state.curr_repo != repo_name:
        st.session_state.curr_repo = repo_name
        st.cache_data.clear()
        
    with st.sidebar:
        with st.spinner(f"Caricamento {circuit}..."):
            raw_df = ml.load_and_prep_data(repo_name)
            if raw_df is not None: st.success("✅ DB Connesso")
            else: st.warning("⚠️ Offline Mode")

    col_p1, col_p2 = st.sidebar.columns(2)
    
    if circuit == "ATP (Uomini)":
        target_p1, target_p2 = "Jannik Sinner", "Carlos Alcaraz"
    else:
        target_p1, target_p2 = "Iga Swiatek", "Aryna Sabalenka"

    player_list = [target_p1, target_p2]
    if raw_df is not None:
        try:
            all_p = pd.concat([raw_df['winner_name'], raw_df['loser_name']]).unique()
            all_p.sort()
            player_list = list(all_p)
        except: pass

    with col_p1:
        default_idx_p1 = player_list.index(target_p1) if target_p1 in player_list else 0
        p1_name = st.selectbox("Giocatore 1", player_list, index=default_idx_p1, key=f"p1_sel_{repo_name}")
        p1_hand = st.selectbox("Mano P1", ["Destra", "Sinistra"])
        p1_bh = st.selectbox("Rovescio P1", ["Due Mani", "Una Mano"])
        
        # Link TennisAbstract Dinamico
        elo_url = "https://www.tennisabstract.com/cgi-bin/leaders_elo.cgi"
        if circuit != "ATP (Uomini)": elo_url += "?f=WTA"
        st.markdown(f"🔗 [Trova Elo su TennisAbstract]({elo_url})")
        elo1 = st.number_input(f"Elo {p1_name}", value=1500, step=10)

    with col_p2:
        default_idx_p2 = player_list.index(target_p2) if target_p2 in player_list else 0
        p2_name = st.selectbox("Giocatore 2", player_list, index=default_idx_p2, key=f"p2_sel_{repo_name}")
        p2_hand = st.selectbox("Mano P2", ["Destra", "Sinistra"])
        p2_bh = st.selectbox("Rovescio P2", ["Due Mani", "Una Mano"])
        
        # Link TennisAbstract Dinamico
        st.markdown(f"🔗 [Trova Elo su TennisAbstract]({elo_url})")
        elo2 = st.number_input(f"Elo {p2_name}", value=1500, step=10)

    st.sidebar.markdown("---")
    
    # --- INFO DELTA ELO ---
    elo_diff = elo1 - elo2
    st.sidebar.info(f"📊 Delta Elo: {int(elo_diff)}")

    surface = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    
    avg_1st, avg_2nd, avg_tot = None, None, None
    if raw_df is not None: avg_1st, avg_2nd, avg_tot = ml.calculate_tour_baselines(raw_df, surface)
    if avg_1st is None: 
        avg_1st = 0.73 if circuit == "ATP (Uomini)" else 0.63
        avg_2nd = 0.52 if circuit == "ATP (Uomini)" else 0.46
    
    st.sidebar.caption(f"Medie Tour ({surface}): 1st {avg_1st:.0%} | 2nd {avg_2nd:.0%}")
    cpi = st.sidebar.slider("CPI (Velocità)", 20, 50, 35)
    ce1, ce2 = st.sidebar.columns(2)
    altitude = ce1.checkbox("Altitudine"); indoor = ce1.checkbox("Indoor"); windy = ce2.checkbox("Vento")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Statistiche (0-100%)")
    p1_data = ml.get_player_prediction(raw_df, p1_name, surface)
    p2_data = ml.get_player_prediction(raw_df, p2_name, surface)
    def val(d, k, fb): return float(d[k]) if d else fb

    if p1_data: st.sidebar.success(f"✅ Dati per {p1_name}")
    else: st.sidebar.warning(f"❌ Dati non trovati per {p1_name}")

    with st.sidebar.expander(f"Stats {p1_name}", expanded=True):
        p1_1in = st.number_input(f"1st In %", 0.0, 100.0, val(p1_data,'1st_in',0.62)*100, format="%.1f", key=f'p1i_{p1_name}') / 100
        p1_1w = st.number_input(f"1st Win %", 0.0, 100.0, val(p1_data,'1st_win',0.74)*100, format="%.1f", key=f'p1w_{p1_name}') / 100
        p1_2w = st.number_input(f"2nd Win %", 0.0, 100.0, val(p1_data,'2nd_win',0.53)*100, format="%.1f", key=f'p1w2_{p1_name}') / 100
        p1_ace = st.number_input(f"Ace %", 0.0, 50.0, val(p1_data,'ace_pct',0.08)*100, format="%.1f", key=f'p1a_{p1_name}') / 100
        p1_df = st.number_input(f"DF %", 0.0, 50.0, val(p1_data,'df_pct',0.03)*100, format="%.1f", key=f'p1d_{p1_name}') / 100
        st.markdown("**Risposta**")
        p1_r1 = st.number_input("Win vs 1st %", 0.0, 100.0, val(p1_data, 'ret_1st_win', 1-avg_1st)*100, format="%.1f", key=f'p1r1_{p1_name}') / 100
        p1_r2 = st.number_input("Win vs 2nd %", 0.0, 100.0, val(p1_data, 'ret_2nd_win', 1-avg_2nd)*100, format="%.1f", key=f'p1r2_{p1_name}') / 100

    if p2_data: st.sidebar.success(f"✅ Dati per {p2_name}")
    else: st.sidebar.warning(f"❌ Dati non trovati per {p2_name}")

    with st.sidebar.expander(f"Stats {p2_name}", expanded=True):
        p2_1in = st.number_input(f"1st In %", 0.0, 100.0, val(p2_data,'1st_in',0.64)*100, format="%.1f", key=f'p2i_{p2_name}') / 100
        p2_1w = st.number_input(f"1st Win %", 0.0, 100.0, val(p2_data,'1st_win',0.73)*100, format="%.1f", key=f'p2w_{p2_name}') / 100
        p2_2w = st.number_input(f"2nd Win %", 0.0, 100.0, val(p2_data,'2nd_win',0.52)*100, format="%.1f", key=f'p2w2_{p2_name}') / 100
        p2_ace = st.number_input(f"Ace %", 0.0, 50.0, val(p2_data,'ace_pct',0.07)*100, format="%.1f", key=f'p2a_{p2_name}') / 100
        p2_df = st.number_input(f"DF %", 0.0, 50.0, val(p2_data,'df_pct',0.04)*100, format="%.1f", key=f'p2d_{p2_name}') / 100
        st.markdown("**Risposta**")
        p2_r1 = st.number_input("Win vs 1st %", 0.0, 100.0, val(p2_data, 'ret_1st_win', 1-avg_1st)*100, format="%.1f", key=f'p2r1_{p2_name}') / 100
        p2_r2 = st.number_input("Win vs 2nd %", 0.0, 100.0, val(p2_data, 'ret_2nd_win', 1-avg_2nd)*100, format="%.1f", key=f'p2r2_{p2_name}') / 100

    st.sidebar.markdown("---")
    
    mot_map = {"Bassa":0.92, "Normale":1.0, "Alta":1.05, "Max":1.10}
    m1 = mot_map[st.sidebar.selectbox(f"Motivazione {p1_name}", list(mot_map.keys()), index=1)]
    m2 = mot_map[st.sidebar.selectbox(f"Motivazione {p2_name}", list(mot_map.keys()), index=1)]
    men1 = st.sidebar.slider(f"Mental {p1_name}", 0.9, 1.1, 1.0)
    men2 = st.sidebar.slider(f"Mental {p2_name}", 0.9, 1.1, 1.0)
    
    col_soldi1, col_soldi2 = st.sidebar.columns(2)
    bankroll = col_soldi1.number_input("Budget (€)", 10, 100000, 1000)
    odds = col_soldi2.number_input(f"Quota {p1_name}", 1.01, 20.0, 1.80)
    
    live_on = st.sidebar.checkbox("LIVE SCORE")
    live_st = None
    if live_on:
        c1,c2 = st.sidebar.columns(2)
        s1=c1.number_input("Set1",0,2,0); s2=c2.number_input("Set2",0,2,0)
        g1=c1.number_input("Gm1",0,6,0); g2=c2.number_input("Gm2",0,6,0)
        pm={0:0,15:1,30:2,40:3,"AD":4}
        pt1=pm[c1.selectbox("Pt1",[0,15,30,40,"AD"])]; pt2=pm[c2.selectbox("Pt2",[0,15,30,40,"AD"])]
        srv_name = st.sidebar.radio("Server Now", [p1_name, p2_name])
        srv = 1 if srv_name==p1_name else 2
        live_st = {'sets_p1':s1,'sets_p2':s2,'games_p1':g1,'games_p2':g2,'pts_p1':pt1,'pts_p2':pt2,'server':srv}
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Micro-Analisi Game")
        if srv == 1:
            base_p = (p1_1w * p1_1in) + (p1_2w * (1-p1_1in))
            prob_markov = TennisMarkov.prob_hold_game(base_p, pt1, pt2)
            prob_mc = TennisMonteCarloEngine.simulate_current_game_mc(base_p, pt1, pt2, men1*m1)
        else:
            base_p = (p2_1w * p2_1in) + (p2_2w * (1-p2_1in))
            prob_markov = TennisMarkov.prob_hold_game(base_p, pt2, pt1)
            prob_mc = TennisMonteCarloEngine.simulate_current_game_mc(base_p, pt2, pt1, men2*m2)
            
        c_m1, c_m2 = st.sidebar.columns(2)
        c_m1.metric("Markov", f"{prob_markov:.1%}")
        c_m2.metric("MC Live", f"{prob_mc:.1%}")
        delta_m = prob_mc - prob_markov
        if abs(delta_m) > 0.05: 
            if delta_m > 0: st.sidebar.success(f"🧠 Boost: +{delta_m:.1%}")
            else: st.sidebar.error(f"🧠 Crollo: {delta_m:.1%}")

    st.title(f"🎾 Tennis Quant Pro | {circuit}")
    st.markdown(f"**{p1_name}** vs **{p2_name}** | {surface} | Best of {sets_to_win*2-1}")
    
    if st.button("🚀 LANCIA SIMULAZIONE COMPLETA"):
        with st.spinner("Calcolo Fisica, Momentum & Log5 Matchup..."):
            conds = {'altitude':altitude, 'indoor':indoor, 'windy':windy}
            
            # --- OLISTIC ELO IMPACT (MANUAL) ---
            diff_elo = elo1 - elo2
            tech_boost = diff_elo / 4000.0 
            mental_impact = 1.0 + (abs(diff_elo) / 2000.0)
            fatigue_impact = 0.005 if abs(diff_elo) > 200 else 0.0
            
            # Applicazione
            p1_1w_used = p1_1w + tech_boost
            p1_2w_used = p1_2w + tech_boost
            
            p1_1w_c, p1_a_c = TennisMath.adjust_stats_for_cpi(p1_1w_used, p1_ace, cpi)
            p1_1w_c, p1_a_c = TennisMath.apply_tactical_adjustments(p1_1w_c, p1_a_c, p1_hand, p2_bh, conds)
            p1_2w_c, _ = TennisMath.adjust_stats_for_cpi(p1_2w_used, 0, cpi)
            p1_2w_c, _ = TennisMath.apply_tactical_adjustments(p1_2w_c, 0, p1_hand, p2_bh, conds)
            
            p2_1w_c, p2_a_c = TennisMath.adjust_stats_for_cpi(p2_1w - tech_boost, p2_ace, cpi)
            p2_1w_c, p2_a_c = TennisMath.apply_tactical_adjustments(p2_1w_c, p2_a_c, p2_hand, p1_bh, conds)
            p2_2w_c, _ = TennisMath.adjust_stats_for_cpi(p2_2w - tech_boost, 0, cpi)
            p2_2w_c, _ = TennisMath.apply_tactical_adjustments(p2_2w_c, 0, p2_hand, p1_bh, conds)
            
            p1_1w_f = TennisMath.log5_matchup(p1_1w_c, p2_r1 * m2, avg_1st)
            p1_2w_f = TennisMath.log5_matchup(p1_2w_c, p2_r2 * m2, avg_2nd)
            p2_1w_f = TennisMath.log5_matchup(p2_1w_c, (p1_r1 + tech_boost) * m1, avg_1st)
            p2_2w_f = TennisMath.log5_matchup(p2_2w_c, (p1_r2 + tech_boost) * m1, avg_2nd)
            
            men1_final = men1 * m1 * (mental_impact if diff_elo > 0 else 1.0)
            men2_final = men2 * m2 * (mental_impact if diff_elo < 0 else 1.0)
            
            fatigue1 = 0.98 + (fatigue_impact if diff_elo > 200 else 0)
            fatigue2 = 0.98 + (fatigue_impact if diff_elo < -200 else 0)

            s1 = {'1st_in':p1_1in, '1st_win':p1_1w_f, '2nd_win':p1_2w_f, 'ace_pct':p1_a_c, 'df_pct':p1_df, 'mental':men1_final, 'fatigue_factor':fatigue1}
            s2 = {'1st_in':p2_1in, '1st_win':p2_1w_f, '2nd_win':p2_2w_f, 'ace_pct':p2_a_c, 'df_pct':p2_df, 'mental':men2_final, 'fatigue_factor':fatigue2}
            
            markov_match_p, markov_set_p = TennisMarkov.get_full_theoretical_prob(s1, s2, sets_to_win*2-1)
            eng = TennisMonteCarloEngine(s1, s2, {'simulations':3000, 'sets_to_win':sets_to_win}, live_st)
            kpi, df = eng.run()
            
            c1,c2,c3,c4 = st.columns(4)
            mc_prob = kpi['win_prob']
            delta = abs(mc_prob - markov_match_p)
            stability = "✅ ALTA" if delta < 0.05 else ("⚠️ MEDIA" if delta < 0.10 else "🚨 BASSA")
            kelly_pct = TennisMath.kelly_criterion(mc_prob, odds, fractional=0.25)
            money_stake = bankroll * kelly_pct
            
            c1.metric("Monte Carlo", f"{mc_prob:.1%}")
            c2.metric("Markov Puro", f"{markov_match_p:.1%}")
            c3.metric("Convergenza", stability, delta=f"Delta: {delta:.1%}")
            c4.metric("Kelly Stake", f"€{money_stake:.2f}")
            
            t1,t2,t3,t4,t5,t6 = st.tabs(["Convergenza", "Partita", "Mercati (U/O)", "Giocatori", "Heatmap", "Insight"])
            
            with t1:
                col_m1, col_m2 = st.columns(2)
                col_m1.info(f"**Monte Carlo:** {mc_prob:.1%}")
                col_m2.warning(f"**Markov Chain:** {markov_match_p:.1%}")
                if delta > 0.10: st.error("Divergenza elevata: I fattori umani stanno ribaltando la statistica.")
            
            with t2:
                cc1,cc2 = st.columns(2)
                cc1.metric("Tie Break", f"{kpi['tb_prob']:.1%}")
                cc2.metric("Rimonta", f"{kpi['comeback_prob']:.1%}")
                st.write(kpi['set_betting'])

            with t3:
                st.subheader("📉 Analisi Under/Over (Fair Odds)")
                m1, m2 = st.columns(2)
                with m1:
                    st.caption("Total Games Match")
                    st.dataframe(TennisMath.generate_market_table(df['tot_games'], "Games"), hide_index=True)
                with m2:
                    st.caption("Total Aces Match")
                    st.dataframe(TennisMath.generate_market_table(df['tot_aces'], "Aces", step=1), hide_index=True)
                
                m3, m4 = st.columns(2)
                with m3:
                    st.caption("Total Breaks Match")
                    st.dataframe(TennisMath.generate_market_table(df['tot_breaks'], "Breaks", step=1), hide_index=True)
                with m4:
                    st.caption("Total Sets (2.5 / 3.5 / 4.5)")
                    st.dataframe(TennisMath.generate_market_table(df['tot_sets'], "Sets", step=1), hide_index=True)

            with t4:
                c_ace, c_brk = st.columns(2)
                c_ace.metric(f"Aces {p1_name}", f"{kpi['avg_aces_p1']:.1f}")
                c_ace.metric(f"Aces {p2_name}", f"{kpi['avg_aces_p2']:.1f}")
                c_brk.metric(f"Breaks {p1_name}", f"{kpi['avg_breaks_p1']:.1f}")
                c_brk.metric(f"Breaks {p2_name}", f"{kpi['avg_breaks_p2']:.1f}")
            with t5:
                hm = df['first_set_score'].value_counts(normalize=True)
                mx = np.zeros((8,8))
                for s,p in hm.items():
                    try: g1,g2=map(int,s.split('-')); 
                    except: continue
                    if g1<8 and g2<8: mx[g1,g2]=p
                fig,ax = plt.subplots(figsize=(5,4))
                sns.heatmap(mx, annot=True, fmt=".0%", cmap="magma", ax=ax, cbar=False)
                ax.invert_yaxis(); ax.set_xlabel(p2_name); ax.set_ylabel(p1_name)
                st.pyplot(fig)
            with t6:
                if p1_hand=="Sinistra" and p2_bh=="Una Mano": st.success("Vantaggio Tattico: Mancino vs 1H-BH")
                if circuit=="ATP (Uomini)" and sets_to_win==3: st.info("Slam Mode: Best of 5 sets")
                if p1_data: st.success("Dati AI caricati.")

if __name__ == "__main__":
    main()
 
