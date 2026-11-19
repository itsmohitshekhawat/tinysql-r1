"""Tests for the scoring code (the reward): no model or API key needed."""

from tinysql import sqlenv
from tinysql.sqlenv import rows_match


def test_extract_sql():
    assert sqlenv.extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert sqlenv.extract_sql("Here you go: select name from t; thanks") == "select name from t"
    assert sqlenv.extract_sql("<think>hmm</think>SELECT 2") == "SELECT 2"


def test_rows_match_ignores_row_order_and_extra_columns():
    gold = [("Ann", 3), ("Bob", 5)]
    assert rows_match([("Bob", 5), ("Ann", 3)], gold)                 # different order
    assert rows_match([(1, "Ann", 3), (2, "Bob", 5)], gold)           # extra id column
    assert rows_match([(3, "Ann"), (5, "Bob")], gold)                 # columns swapped


def test_rows_match_needs_the_columns_the_question_asks_for():
    gold = [("Ann", 3), ("Bob", 5)]                                   # reference returns (name, count)
    assert not rows_match([("Ann",), ("Bob",)], gold)                 # both needed (default): names alone are not enough
    assert rows_match([("Ann",), ("Bob",)], gold, need=[0])           # question asks only "which people": names are enough
    assert not rows_match([(3,), (5,)], gold, need=[0])               # the needed column is missing
    assert rows_match([(3, "Ann"), (5, "Bob")], gold, need=[0])       # extra column is fine


def test_rows_match_rejects_wrong_answers():
    gold = [("Ann", 3), ("Bob", 5)]
    assert not rows_match([("Ann", 3)], gold)                         # missing row
    assert not rows_match([("Ann", 5), ("Bob", 3)], gold)             # values paired wrongly
    assert not rows_match([("Ann", 3), ("Cid", 5)], gold)             # wrong name
    assert not rows_match([(7,), (8,)], gold, need=[0])               # unrelated column


def test_every_question_has_valid_needed_columns():
    import json

    data = json.loads((sqlenv.ROOT / "data" / "questions.json").read_text())
    for part in data.values():
        for x in part:
            width = len(sqlenv.columns(x["db"], x["sql"]))
            assert x["need"] and all(0 <= i < width for i in x["need"])
            assert sqlenv.same_result(x["db"], x["sql"], x["sql"], x["need"])[0]  # the reference passes its own check


def test_run_is_read_only_and_bounded():
    assert sqlenv.run("shop", "DROP TABLE customers")[1]
    assert sqlenv.run("shop", "SELECT nope FROM customers")[1]
    rows, err = sqlenv.run("shop", "SELECT COUNT(*) FROM customers")
    assert err is None and rows == [(25,)]
    _, err = sqlenv.run("shop", "SELECT COUNT(*) FROM orders a, orders b, orders c, orders d, orders e")
    assert err  # cancelled: too many steps


def test_reward_levels():
    gold = "SELECT name FROM customers WHERE city = 'Berlin'"
    assert sqlenv.reward("shop", gold, gold) == 1.0
    assert sqlenv.reward("shop", "SELECT id, name FROM customers WHERE city = 'Berlin'", gold) == 1.0
    assert sqlenv.reward("shop", "SELECT name FROM customers", gold) == 0.2
    assert sqlenv.reward("shop", "SELECT id FROM customers WHERE city = 'Berlin'", gold) == 0.2  # right rows, wrong column
    assert sqlenv.reward("shop", "I think the answer is 42", gold) == 0.0
