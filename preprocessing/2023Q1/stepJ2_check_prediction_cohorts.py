import pandas as pd

path = r"Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"

usecols = [
    "loan_identifier",
    "loan_age",
    "is_60plus_delinquent"
]

print("Loading temporal data...")

df = pd.read_csv(
    path,
    usecols=usecols
)

df["loan_age"] = pd.to_numeric(
    df["loan_age"],
    errors="coerce"
)

df = df.dropna(
    subset=["loan_age"]
)

df["loan_age"] = df["loan_age"].astype(int)

print("Loaded.")
print("Rows:", len(df))
print("Unique loans:", df["loan_identifier"].nunique())


def check_cohort(observation_age):

    print("\n======================================")
    print("OBSERVATION AGE:", observation_age)
    print("======================================")

    # Loans available exactly at observation age
    obs_ids = set(
        df.loc[
            df["loan_age"] == observation_age,
            "loan_identifier"
        ]
    )

    print(
        "Loans at exact observation age:",
        len(obs_ids)
    )

    # Require observation age + next 6 months
    required_sets = []

    for age in range(
        observation_age,
        observation_age + 7
    ):

        ids = set(
            df.loc[
                df["loan_age"] == age,
                "loan_identifier"
            ]
        )

        required_sets.append(ids)

    full_followup_ids = set.intersection(
        *required_sets
    )

    print(
        "Loans with full 6-month follow-up:",
        len(full_followup_ids)
    )

    # Remove loans already 60+ delinquent
    already_60plus = set(
        df.loc[
            (df["loan_age"] == observation_age)
            &
            (df["is_60plus_delinquent"] == 1),
            "loan_identifier"
        ]
    )

    final_eligible = (
        full_followup_ids
        - already_60plus
    )

    print(
        "Already 60+ at observation:",
        len(
            full_followup_ids
            & already_60plus
        )
    )

    print(
        "FINAL ELIGIBLE LOANS:",
        len(final_eligible)
    )


for age in [12, 18, 24]:
    check_cohort(age)