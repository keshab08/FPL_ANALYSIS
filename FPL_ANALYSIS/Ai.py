import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import requests

# Fetch data from API
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player stats
players = pd.DataFrame(data["elements"])

# Select relevant columns
players = players[["id", "web_name", "team", "now_cost", "minutes", "goals_scored",
                   "assists", "clean_sheets", "bonus", "form", "threat",
                   "influence", "creativity", "expected_goals", "expected_assists",
                   "selected_by_percent", "points_per_game", "total_points"]]

print("Available players:", players["web_name"].unique())

# Rename columns
players.rename(columns={"now_cost": "value", "expected_goals": "xG", "expected_assists": "xA"}, inplace=True)

# Add Fixture Difficulty Rating (Random for now)
players["FDR"] = np.random.randint(1, 5, players.shape[0])

# Use `points_per_game` as the target
players["next_gameweek_points"] = players["points_per_game"]

# Save dataset
players.to_csv("fpl_player_data.csv", index=False)

# Feature Selection
features = ["minutes", "goals_scored", "assists", "clean_sheets", "bonus", "form",
            "value", "threat", "influence", "creativity", "xG", "xA", "FDR"]
X = players[features]
y = players["next_gameweek_points"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae:.2f}")

# Save the trained model
joblib.dump(model, "fpl_ai_model.pkl")


# Fetch player stats
# Fetch player stats
def get_player_stats(player_name):
    """Fetches FPL stats for a given player."""

    # Reload FPL data
    response = requests.get(FPL_API_URL)
    data = response.json()
    players = pd.DataFrame(data["elements"])

    # Print all available players (for debugging)



    # Find the player (case-insensitive)
    player = players[players["web_name"].str.lower() == player_name.lower()]

    if player.empty:
        print(f"⚠️ Player '{player_name}' not found! Check name spelling.")
        return None

    # Explicitly create a copy of the player DataFrame to avoid SettingWithCopyWarning
    player = player.copy()

    # Rename columns to match the training data
    player.rename(columns={"now_cost": "value", "expected_goals": "xG", "expected_assists": "xA"}, inplace=True)

    # Extract stats
    stats = {
        "minutes": player.iloc[0]["minutes"],
        "goals_scored": player.iloc[0]["goals_scored"],
        "assists": player.iloc[0]["assists"],
        "clean_sheets": player.iloc[0]["clean_sheets"],
        "bonus": player.iloc[0]["bonus"],
        "form": float(player.iloc[0]["form"]),
        "value": player.iloc[0]["value"],
        "threat": player.iloc[0]["threat"],
        "influence": player.iloc[0]["influence"],
        "creativity": player.iloc[0]["creativity"],
        "xG": player.iloc[0]["xG"],
        "xA": player.iloc[0]["xA"],
        "FDR": np.random.randint(1, 5),
    }
    return list(stats.values())

# Predict FPL points
def predict_fpl_points(player_name):
    """Predicts points using the trained model."""

    player_stats = get_player_stats(player_name)
    if player_stats is None:
        return None

    # Load trained model
    model = joblib.load("fpl_ai_model.pkl")

    # Convert to DataFrame
    feature_names = ["minutes", "goals_scored", "assists", "clean_sheets", "bonus", "form",
                     "value", "threat", "influence", "creativity", "xG", "xA", "FDR"]
    player_df = pd.DataFrame([player_stats], columns=feature_names)

    # Predict and return result
    predicted_points = model.predict(player_df)[0]
    print(f"🔮 Predicted Points for {player_name}: {predicted_points:.2f}")
    return predicted_points


# Example Usage
predict_fpl_points("Haaland")
predict_fpl_points("M.Salah")
predict_fpl_points("Munetsi")