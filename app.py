
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv(
    "food delivery analytics.csv",
    sep=None,
    engine="python"
)

# Create transaction basket
basket = (
    df.groupby(["OrderID", "FoodItem"])["Quantity"]
      .sum()
      .unstack(fill_value=0)
)

# Convert quantity to 0 and 1
basket = basket.map(lambda x: 1 if x > 0 else 0)

# Generate frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.03,
    use_colnames=True
)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.20
)

# Create FastAPI application
app = FastAPI(
    title="Food Delivery Recommendation API",
    description="Food recommendation system using Apriori",
    version="1.0"
)


class FoodRequest(BaseModel):
    food_item: str
    number_of_recommendations: int = 5


@app.get("/")
def home():
    return {
        "message": "Food Delivery Recommendation API is running"
    }


@app.post("/recommend")
def recommend(request: FoodRequest):

    food_item = request.food_item
    number_of_recommendations = request.number_of_recommendations

    # Check food item
    if food_item not in basket.columns:
        return {
            "food_item": food_item,
            "recommendations": [],
            "message": "Food item not found"
        }

    # Find matching rules
    recommendations = rules[
        rules["antecedents"].map(
            lambda x: food_item in x
        )
    ].copy()

    if recommendations.empty:
        return {
            "food_item": food_item,
            "recommendations": []
        }

    # Sort by lift and confidence
    recommendations = recommendations.sort_values(
        ["lift", "confidence"],
        ascending=False
    )

    result = []

    for _, row in recommendations.iterrows():

        for item in row["consequents"]:

            if item != food_item and item not in result:
                result.append(item)

            if len(result) >= number_of_recommendations:
                break

        if len(result) >= number_of_recommendations:
            break

    return {
        "food_item": food_item,
        "recommendations": result
    }
