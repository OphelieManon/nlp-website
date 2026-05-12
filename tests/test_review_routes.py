"""Integration tests for /product/<id>/review."""

import pandas as pd


def _valid_form(predicted_label=None):
    """Build a form dict that passes validation."""
    data = {
        "title": "Lovely product",
        "rating": "5",
        "review_text": "I am really happy with this purchase, will buy again.",
    }
    if predicted_label is not None:
        data["predicted_label"] = str(predicted_label)
    return data


def test_get_review_form_renders_empty(client):
    resp = client.get("/product/103/review")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'action="/product/103/review"' in body
    assert 'name="title"' in body
    assert 'name="rating"' in body
    assert 'name="review_text"' in body


def test_post_predict_step_shows_prediction_banner(client):
    resp = client.post("/product/103/review", data=_valid_form())
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # The prediction banner must show one of these phrases
    assert ("Would buy" in body) or ("Would not buy" in body)
    # The override radio must be present
    assert 'name="predicted_label"' in body
    # Form fields are prefilled
    assert "Lovely product" in body


def test_post_empty_title_renders_error(client):
    form = _valid_form()
    form["title"] = "   "
    resp = client.post("/product/103/review", data=form)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "title" in body.lower()
    # No banner — validation failed before predict
    assert "predicted_label" not in body or "Would" not in body


def test_post_rating_out_of_range_renders_error(client):
    form = _valid_form()
    form["rating"] = "9"
    resp = client.post("/product/103/review", data=form)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "rating" in body.lower()


def test_post_whitespace_review_text_renders_error(client):
    form = _valid_form()
    form["review_text"] = "    "
    resp = client.post("/product/103/review", data=form)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "review" in body.lower()


def test_post_commit_redirects_and_review_visible(client):
    resp = client.post(
        "/product/103/review",
        data=_valid_form(predicted_label=1),
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/product/103" in resp.headers["Location"]

    # Now GET the detail page and look for the new review
    resp2 = client.get("/product/103")
    body = resp2.get_data(as_text=True)
    assert "Lovely product" in body
    assert "I am really happy with this purchase" in body


def test_new_review_appears_above_existing_ones(client):
    """Spec §7.7: new review must appear at the top of the reviews list."""
    # Product 103 has a pre-existing review "Would not recommend this product."
    client.post("/product/103/review", data=_valid_form(predicted_label=1))
    body = client.get("/product/103").get_data(as_text=True)
    new_pos = body.find("Lovely product")
    existing_pos = body.find("Would not recommend this product.")
    assert new_pos != -1
    assert existing_pos != -1
    assert new_pos < existing_pos, "new review should appear before existing reviews"


def test_post_commit_persists_to_csv(client, tmp_path):
    client.post("/product/103/review", data=_valid_form(predicted_label=1))
    # The conftest's client fixture uses tmp_path/data; reviews.csv is there
    csv = tmp_path / "data" / "reviews.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    titles = df["title"].astype(str).tolist()
    assert "Lovely product" in titles


def test_post_review_for_unknown_product_returns_404(client):
    resp = client.post("/product/99999/review", data=_valid_form())
    assert resp.status_code == 404
