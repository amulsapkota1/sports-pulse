import pandas as pd
import json
import networkx as nx
import file_read
# ---------------- Utility Methods ----------------
def parse_json_safe(x):
    """Safely parse JSON strings into dicts."""
    try:
        return json.loads(x) if pd.notnull(x) and str(x).strip() != '' else {}
    except Exception:
        return {}


# ---------------- Data Processing ----------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: fix types, drop duplicates, parse JSON fields."""
    # Convert dates
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['accessed_date'] = pd.to_datetime(df['accessed_date'], errors='coerce')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Parse JSON fields
    df['stats'] = df['stats_json'].apply(parse_json_safe)
    df['extras'] = df['extras_json'].apply(parse_json_safe)

    # Extract selected stats
    df['fighter_a_record_at_fight'] = df['stats'].apply(lambda x: x.get('fighter_a_record_at_fight'))
    df['fighter_b_record_at_fight'] = df['stats'].apply(lambda x: x.get('fighter_b_record_at_fight'))

    df['fighter_a_age'] = df['extras'].apply(lambda x: x.get('fighter_a_age_at_fight_years'))
    df['fighter_b_age'] = df['extras'].apply(lambda x: x.get('fighter_b_age_at_fight_years'))

    df['fighter_a_height_cm'] = df['extras'].apply(lambda x: x.get('fighter_a_height_cm'))
    df['fighter_b_height_cm'] = df['extras'].apply(lambda x: x.get('fighter_b_height_cm'))

    return df


def filter_rabindra_fights(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for Rabindra Dhant fights."""
    df_rabindra = df[(df['fighter_a'] == 'Rabindra Dhant') | (df['fighter_b'] == 'Rabindra Dhant')]

    def rabindra_outcome(row):
        if row['fighter_a'] == 'Rabindra Dhant':
            return row['outcome']
        elif row['fighter_b'] == 'Rabindra Dhant':
            return 'Loss' if str(row['outcome']).lower() == 'win' else 'Win'
        return None

    df_rabindra['rabindra_outcome'] = df_rabindra.apply(rabindra_outcome, axis=1)
    return df_rabindra


def summarize_rabindra(df_rabindra: pd.DataFrame):
    """Print summary statistics of Rabindra Dhant's career."""
    print("\nRabindra Dhant fight records count:", len(df_rabindra))
    print("\nRabindra Dhant Fight Outcomes:")
    print(df_rabindra['rabindra_outcome'].value_counts())

    print("\nWin methods by Rabindra Dhant:")
    print(df_rabindra[df_rabindra['rabindra_outcome'] == 'Win']['method'].value_counts())


# ---------------- Visualization ----------------
def build_mindmap() -> nx.DiGraph:
    """Build a mindmap graph for Rabindra Dhant's career highlights."""
    background = [
        "Born: November 30, 1998, Bajhang District, Nepal",
        "Early life: Manual laborer in India (Pithoragarh, New Delhi)",
        "Initial training: Karate, then transitioned to MMA",
        "Nickname: The Tiger of Bajhang",
        "Nationality: Nepalese (refused Indian citizenship offer)"
    ]

    professional_career = [
        "Record: 9 wins, 1 loss (7 KO/TKO, 1 Submission, 1 Decision)",
        "Notable fights: MFN Bantamweight Championship win vs Chungreng Koren",
        "Fight locations: India, China, Thailand, Nepal",
        "Current streak: 3 wins"
    ]

    training_support = [
        "Current gym: Soma Fight Club, Bali (Indonesia)",
        "Previous training: Lock N Roll MMA, Kathmandu",
        "Mentorship: Coach Diwiz Piya Lama",
        "Sponsorship & finances: Supported by Nutrition Fit Nepali and Latido"
    ]

    challenges_mindset = [
        "Height initially self-conscious for volleyball",
        "Citizenship barriers for international fights",
        "Financial struggles, lack of institutional support",
        "Family pressure to find stable job",
        "Intense training (2-3 times/day), body aches normal",
        "Mental toughness & discipline emphasized by coaches",
        "Focus on fighting itself to handle pressure",
        "Avoids social media distractions during fight week",
        "Finds peace in Nepali village vlogs",
        "Prefers not to watch other fights before bouts",
        "Goal: UFC World Champion (target age 33-34)"
    ]

    impact_recognition = [
        "Widespread recognition in Nepal (public, media, political leaders)",
        "Milestone for combat sports in Nepal",
        "Congratulated by Prime Minister K.P. Sharma Oli and others",
        "Inspires young Nepali athletes",
        "Helped popularize MMA in Nepal",
        "Role model: humble, disciplined, respectful"
    ]

    mma_in_nepal = [
        "Growing popularity over last 5 years",
        "Nepal Warriors Championship (NWC): platform for local fighters",
        "Challenges: dearth of players, lack of proper facilities, minimal government support",
        "High prices for gyms/ facilities in Kathmandu (~4.5-5K NPR/month)",
        "No official governing body for MMA as of 2019",
        "Emphasizes discipline, perseverance, and hard work"
    ]

    controversy = [
        "Warriors Cove (Chungreng's team) claimed Koren had leg issues before MFN 17",
        "Coach Mike (Dhant's coach) strongly denied excuses, blaming Indian MMA culture",
        "Highlighted Dhant fought with an MCL tear, not wrestling for 6 weeks",
        "Warriors Cove criticized for inconsistent weight management",
        "Dhant outclassed opponent in wrestling, striking, grappling",
        "No rematch deserved due to Warriors Cove's lack of accountability",
        "Emphasis on humility and learning from losses for growth"
    ]

    # Build Graph
    G = nx.DiGraph()
    G.add_node("Rabindra Dhant: Nepali MMA Champion")

    for branch, items in zip(
        ["Background", "Professional Career", "Training & Support", "Challenges & Mindset",
         "Impact & Recognition", "MMA in Nepal", "Controversy"],
        [background, professional_career, training_support, challenges_mindset,
         impact_recognition, mma_in_nepal, controversy]
    ):
        G.add_node(branch)
        G.add_edge("Rabindra Dhant: Nepali MMA Champion", branch)
        for item in items:
            label = item if len(item) <= 60 else item[:60] + "..."
            G.add_node(label)
            G.add_edge(branch, label)

    return G


# ---------------- Main Execution ----------------
def main():

    df = file_read.readFile()

    print("Dataframe info:")
    print(df.info())
    print("\nSample records:")
    print(df.head())

    df_clean = clean_data(df)

    print("\nMissing values per column:")
    print(df_clean.isnull().sum())

    df_rabindra = filter_rabindra_fights(df_clean)
    summarize_rabindra(df_rabindra)

    G = build_mindmap()
    print("\nMindmap graph built with", len(G.nodes), "nodes and", len(G.edges), "edges.")
