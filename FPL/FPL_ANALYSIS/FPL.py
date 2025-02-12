import requests


TEAM_ID = 5650650  # Replace with your actual FPL team ID
url = f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    # Extract leagues
    classic_leagues = data["leagues"]["classic"]
    h2h_leagues = data["leagues"]["h2h"]

    # Print Classic Leagues
    print("\n🏆 Classic Leagues:")
    for league in classic_leagues:
        print(f"- {league['name']} (Rank: {league['entry_rank']})")

    # Print Head-to-Head Leagues
    print("\n⚔️ Head-to-Head Leagues:")
    for league in h2h_leagues:
        print(f"- {league['name']} (Rank: {league['entry_rank']})")

else:
    print(f"Error: {response.status_code}")


LEAGUE_ID = 23876  # Replace with your actual League ID
url = f"https://fantasy.premierleague.com/api/leagues-classic/{LEAGUE_ID}/standings/"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    # Extract league name
    league_name = data["league"]["name"]
    print(f"\n🏆 League: {league_name}\n")

    # Extract player standings
    standings = data["standings"]["results"]

    print("📋 Players in the League:")
    for player in standings:
        print(f"- {player['player_name']} ({player['entry_name']}) - Points: {player['total']}")

else:
    print(f"Error: {response.status_code}")
