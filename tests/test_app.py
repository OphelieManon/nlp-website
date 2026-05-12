from app.nlp.data_loader import load_products


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_shows_product_count(client):
    products = load_products()
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert f"{len(products)} products" in body


def test_index_shows_first_product_name(client):
    products = load_products()
    first_name = products.iloc[0]["product_name"]
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert first_name in body


def test_index_includes_navbar_search_form(client):
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert 'name="q"' in body
    assert 'action="/search"' in body
