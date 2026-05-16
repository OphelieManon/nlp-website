from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

from app.nlp.data_loader import load_product_stats, load_products, load_reviews, load_user_reviews

admin_bp = Blueprint("admin", __name__)

# Change these to secure the admin panel.
_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD = "admin123"


def _login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.admin_login"))
        return f(*args, **kwargs)
    return decorated


# ==============================
# LOGIN / LOGOUT
# ==============================

@admin_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == _ADMIN_USERNAME and password == _ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin.admin_dashboard"))
        error = "Invalid username or password."
    return render_template("admin_login.html", error=error)


@admin_bp.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin.admin_login"))


# ==============================
# DASHBOARD
# ==============================

@admin_bp.route("/admin")
@_login_required
def admin_dashboard():
    products = load_products()
    reviews = load_user_reviews()

    overrides_count = 0
    if "final_label" in reviews.columns and "predicted_label" in reviews.columns:
        overrides_count = int(
            (reviews["predicted_label"] != reviews["final_label"]).sum()
        )

    return render_template(
        "admin_dashboard.html",
        total_products=len(products),
        total_reviews=len(reviews),
        overrides_count=overrides_count,
    )


# ==============================
# STATISTICS
# ==============================

@admin_bp.route("/admin/statistics")
@_login_required
def statistics():
    reviews = load_user_reviews()
    total = len(reviews)
    avg_rating = round(reviews["rating"].mean(), 2) if total else 0
    buy_count = int((reviews["predicted_label"] == 1).sum())
    not_buy_count = int((reviews["predicted_label"] == 0).sum())
    buy_pct = round(buy_count / total * 100, 1) if total else 0
    not_buy_pct = round(not_buy_count / total * 100, 1) if total else 0

    product_stats = load_product_stats()
    top10 = (
        product_stats[product_stats["product_rating_count"] >= 5]
        .sort_values(
            ["avg_product_rating", "product_rating_count", "product_id"],
            ascending=[False, False, True],
        )
        .head(10)
        .reset_index(drop=True)
        .rename(columns={
            "avg_product_rating":   "avg",
            "product_rating_count": "count",
        })
    )
    top_products = top10.to_dict(orient="records")

    return render_template(
        "admin_statistics.html",
        avg_rating=avg_rating,
        buy_count=buy_count,
        not_buy_count=not_buy_count,
        buy_pct=buy_pct,
        not_buy_pct=not_buy_pct,
        total=total,
        top_products=top_products,
    )


# ==============================
# REVIEW MODERATION
# ==============================

_REVIEWS_PER_PAGE = 20


def _paginate(records: list, per_page: int = _REVIEWS_PER_PAGE):
    """Slice a list of dicts for the current ?page= and return pagination vars."""
    total = len(records)
    pages = max(1, (total + per_page - 1) // per_page)
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    return records[start:start + per_page], page, pages, total

@admin_bp.route("/admin/reviews")
@_login_required
def review_moderation():
    records = load_user_reviews().to_dict(orient="records")
    page_records, page, pages, total = _paginate(records)
    return render_template(
        "review_moderation.html",
        heading="Reviews",
        reviews=page_records,
        page=page, pages=pages, total=total,
    )


# ==============================
# OVERRIDDEN LABELS
# ==============================

@admin_bp.route("/admin/overrides")
@_login_required
def overridden_reviews():
    reviews = load_user_reviews()
    if "final_label" not in reviews.columns:
        records = []
    else:
        records = reviews[
            reviews["predicted_label"] != reviews["final_label"]
        ].to_dict(orient="records")
    page_records, page, pages, total = _paginate(records)
    return render_template(
        "review_moderation.html",
        heading="Label Corrections",
        reviews=page_records,
        page=page, pages=pages, total=total,
    )




