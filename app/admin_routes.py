from flask import Blueprint, render_template

from app.nlp.data_loader import load_products, load_reviews

admin_bp = Blueprint("admin", __name__)


# ==============================
# DASHBOARD
# ==============================

@admin_bp.route("/admin")
def admin_dashboard():

    products = load_products()
    reviews = load_reviews()

    total_products = products["product_id"].nunique()

    total_reviews = len(reviews)

    avg_rating = round(reviews["rating"].mean(), 2)

    buy_count = len(
        reviews[
            reviews["predicted_label"] == 1
        ]
    )

    not_buy_count = len(
        reviews[
            reviews["predicted_label"] == 0
        ]
    )

    return render_template(
        "admin_dashboard.html",
        total_products=total_products,
        total_reviews=total_reviews,
        avg_rating=avg_rating,
        buy_count=buy_count,
        not_buy_count=not_buy_count
    )


# ==============================
# REVIEW MODERATION
# ==============================

@admin_bp.route("/admin/reviews")
def review_moderation():

    reviews = load_reviews()

    review_list = reviews.to_dict(orient="records")

    return render_template(
        "review_moderation.html",
        reviews=review_list
    )


# ==============================
# OVERRIDDEN LABELS
# ==============================

@admin_bp.route("/admin/overrides")
def overridden_reviews():

    reviews = load_reviews()

    # only if final_label exists
    if "final_label" not in reviews.columns:
        overridden = []
    else:
        overridden = reviews[
            reviews["predicted_label"] != reviews["final_label"]
        ].to_dict(orient="records")

    return render_template(
        "review_moderation.html",
        reviews=overridden
    )


# ==============================
# INCONSISTENT PREDICTIONS
# ==============================

@admin_bp.route("/admin/inconsistent")
def inconsistent_predictions():

    reviews = load_reviews()

    inconsistent = reviews[
        (
            (reviews["rating"] >= 4) &
            (reviews["predicted_label"] == 0)
        )
        |
        (
            (reviews["rating"] <= 2) &
            (reviews["predicted_label"] == 1)
        )
    ]

    inconsistent = inconsistent.to_dict(orient="records")

    return render_template(
        "review_moderation.html",
        reviews=inconsistent
    )


# ==============================
# ANALYTICS
# ==============================

@admin_bp.route("/admin/analytics")
def analytics():

    products = load_products()
    reviews = load_reviews()

    # highest rated products
    # Highest rated products
    highest_rated = (
        reviews
        .groupby("product_id")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    # buy / not buy distribution
    prediction_dist = (
        reviews["predicted_label"]
        .value_counts()
    )

    # # brand counts from products dataset
    # top_brands = (
    #     products["brand"]
    #     .value_counts()
    #     .head(10)
    # )

    return render_template(
        "analytics.html",

        # brand_labels=top_brands.index.tolist(),
        # brand_values=top_brands.values.tolist(),

        product_labels=highest_rated.index.tolist(),
        product_values=highest_rated.values.tolist(),

        prediction_labels=prediction_dist.index.tolist(),
        prediction_values=prediction_dist.values.tolist()
    )