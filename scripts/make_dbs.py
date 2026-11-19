"""Create the 8 small SQLite databases used for training and testing (deterministic: same seed, same data).

Six databases are used for training. Two (movies, sports) are held out completely: the model never sees their
schema during training, so they test whether it learned SQL in general or just memorised six schemas.

Usage: python scripts/make_dbs.py   -> data/dbs/<name>.sqlite and data/schemas.json
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBS = ROOT / "data" / "dbs"
HELD_OUT = ["movies", "sports"]

FIRST = ["Aarav", "Maya", "Liam", "Sofia", "Noah", "Zara", "Ethan", "Priya", "Lucas", "Amara", "Kenji", "Elena",
         "Omar", "Hana", "Diego", "Freya", "Ravi", "Chloe", "Mateo", "Ines", "Tariq", "Nora", "Felix", "Anya"]
LAST = ["Sharma", "Garcia", "Muller", "Rossi", "Tanaka", "Okafor", "Novak", "Silva", "Khan", "Larsen", "Dubois",
        "Kowalski", "Mehta", "Haddad", "Jensen", "Moreau", "Costa", "Nakamura", "Ivanov", "Reyes"]
CITIES = ["Berlin", "Madrid", "Mumbai", "Tokyo", "Lagos", "Toronto", "Lisbon", "Vienna", "Cairo", "Oslo"]
COUNTRIES = ["Germany", "Spain", "India", "Japan", "Nigeria", "Canada", "Portugal", "Austria", "Egypt", "Norway"]


def names(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in out:
            out.append(name)
    return out


def date(rng: random.Random, y0: int = 2022, y1: int = 2024) -> str:
    return f"{rng.randint(y0, y1)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def shop(rng):
    ddl = """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_year INTEGER);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, order_date TEXT);"""
    cats = {"Electronics": ["Headphones", "Keyboard", "Monitor", "Webcam", "Speaker"], "Home": ["Lamp", "Kettle", "Blanket", "Vase"],
            "Sports": ["Yoga Mat", "Dumbbells", "Football", "Tennis Racket"], "Books": ["Novel", "Cookbook", "Atlas"]}
    products = [(n, c, round(rng.uniform(8, 320), 2)) for c, ns in cats.items() for n in ns]
    rows = {"customers": [(i + 1, n, rng.choice(CITIES), rng.randint(2018, 2024)) for i, n in enumerate(names(rng, 25))],
            "products": [(i + 1, *p) for i, p in enumerate(products)]}
    rows["orders"] = [(i + 1, rng.randint(1, 25), rng.randint(1, len(products)), rng.randint(1, 5), date(rng)) for i in range(60)]
    return ddl, rows


def school(rng):
    ddl = """CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, grade_level INTEGER, age INTEGER);
CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, department TEXT, credits INTEGER);
CREATE TABLE enrollments (id INTEGER PRIMARY KEY, student_id INTEGER REFERENCES students(id), course_id INTEGER REFERENCES courses(id), score INTEGER);"""
    courses = [("Algebra", "Math"), ("Geometry", "Math"), ("Statistics", "Math"), ("Biology", "Science"), ("Chemistry", "Science"),
               ("Physics", "Science"), ("World History", "Humanities"), ("Literature", "Humanities"), ("Programming", "Computing"),
               ("Databases", "Computing")]
    rows = {"students": [(i + 1, n, g := rng.randint(9, 12), g + rng.randint(5, 6)) for i, n in enumerate(names(rng, 30))],
            "courses": [(i + 1, t, d, rng.choice([2, 3, 4])) for i, (t, d) in enumerate(courses)]}
    rows["enrollments"] = [(i + 1, rng.randint(1, 30), rng.randint(1, 10), rng.randint(40, 100)) for i in range(80)]
    return ddl, rows


def library(rng):
    ddl = """CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER REFERENCES authors(id), year INTEGER, genre TEXT, pages INTEGER);
CREATE TABLE loans (id INTEGER PRIMARY KEY, book_id INTEGER REFERENCES books(id), member_name TEXT, loan_date TEXT, returned INTEGER);"""
    words = ["Silent", "Golden", "Hidden", "Last", "Burning", "Winter", "Glass", "Distant", "Broken", "Paper"]
    nouns = ["River", "Garden", "Empire", "Letter", "Harbor", "Mirror", "Orchard", "Signal", "Kingdom", "Compass"]
    titles = sorted({f"The {rng.choice(words)} {rng.choice(nouns)}" for _ in range(60)})[:28]
    rows = {"authors": [(i + 1, n, rng.choice(COUNTRIES)) for i, n in enumerate(names(rng, 12))],
            "books": [(i + 1, t, rng.randint(1, 12), rng.randint(1950, 2023), rng.choice(["Mystery", "Fantasy", "History", "Science", "Romance"]),
                       rng.randint(120, 780)) for i, t in enumerate(titles)]}
    members = names(rng, 15)
    rows["loans"] = [(i + 1, rng.randint(1, len(titles)), rng.choice(members), date(rng), rng.choice([0, 1, 1])) for i in range(50)]
    return ddl, rows


def hospital(rng):
    ddl = """CREATE TABLE doctors (id INTEGER PRIMARY KEY, name TEXT, specialty TEXT, years_experience INTEGER);
CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT);
CREATE TABLE appointments (id INTEGER PRIMARY KEY, doctor_id INTEGER REFERENCES doctors(id), patient_id INTEGER REFERENCES patients(id), date TEXT, fee REAL);"""
    rows = {"doctors": [(i + 1, n, rng.choice(["Cardiology", "Dermatology", "Pediatrics", "Neurology", "Orthopedics"]), rng.randint(1, 35))
                        for i, n in enumerate(names(rng, 12))],
            "patients": [(i + 1, n, rng.randint(2, 90), rng.choice(CITIES)) for i, n in enumerate(names(rng, 30))]}
    rows["appointments"] = [(i + 1, rng.randint(1, 12), rng.randint(1, 30), date(rng), float(rng.choice([40, 60, 80, 120, 150, 200])))
                            for i in range(70)]
    return ddl, rows


def flights(rng):
    ddl = """CREATE TABLE airlines (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE airports (code TEXT PRIMARY KEY, city TEXT, country TEXT);
CREATE TABLE flights (id INTEGER PRIMARY KEY, airline_id INTEGER REFERENCES airlines(id), origin TEXT REFERENCES airports(code), destination TEXT REFERENCES airports(code), duration_min INTEGER, price REAL);"""
    airports = [("BER", "Berlin", "Germany"), ("MAD", "Madrid", "Spain"), ("BOM", "Mumbai", "India"), ("HND", "Tokyo", "Japan"),
                ("LOS", "Lagos", "Nigeria"), ("YYZ", "Toronto", "Canada"), ("LIS", "Lisbon", "Portugal"), ("VIE", "Vienna", "Austria")]
    airlines = [("Skyward", "Germany"), ("Iberia Sol", "Spain"), ("IndiJet", "India"), ("Nippon Wings", "Japan"), ("Maple Air", "Canada"),
                ("Atlantic Blue", "Portugal")]
    rows = {"airlines": [(i + 1, *a) for i, a in enumerate(airlines)], "airports": airports, "flights": []}
    for i in range(60):
        o, d = rng.sample([a[0] for a in airports], 2)
        rows["flights"].append((i + 1, rng.randint(1, 6), o, d, rng.randint(55, 780), round(rng.uniform(45, 990), 2)))
    return ddl, rows


def hr(rng):
    ddl = """CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, location TEXT);
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER REFERENCES departments(id), salary INTEGER, hire_year INTEGER, title TEXT);
CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER REFERENCES departments(id), budget INTEGER);"""
    deps = ["Engineering", "Sales", "Marketing", "Finance", "Support", "Research"]
    rows = {"departments": [(i + 1, d, rng.choice(CITIES)) for i, d in enumerate(deps)],
            "employees": [(i + 1, n, rng.randint(1, 6), rng.randrange(32000, 140000, 1000), rng.randint(2010, 2024),
                           rng.choice(["Analyst", "Manager", "Engineer", "Director", "Associate"])) for i, n in enumerate(names(rng, 40))],
            "projects": [(i + 1, f"Project {w}", rng.randint(1, 6), rng.randrange(20000, 900000, 5000))
                         for i, w in enumerate(["Atlas", "Beacon", "Comet", "Delta", "Echo", "Falcon", "Granite", "Harbor", "Iris", "Jade",
                                                "Kite", "Lumen"])]}
    return ddl, rows


def movies(rng):
    ddl = """CREATE TABLE directors (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE movies (id INTEGER PRIMARY KEY, title TEXT, director_id INTEGER REFERENCES directors(id), year INTEGER, genre TEXT, rating REAL);
CREATE TABLE reviews (id INTEGER PRIMARY KEY, movie_id INTEGER REFERENCES movies(id), reviewer TEXT, stars INTEGER);"""
    a = ["Midnight", "Crimson", "Electric", "Forgotten", "Rising", "Velvet", "Iron", "Lunar", "Wild", "Quiet"]
    b = ["Tide", "Echo", "Horizon", "Promise", "Circuit", "Voyage", "Shadow", "Anthem", "Frontier", "Parade"]
    titles = sorted({f"{rng.choice(a)} {rng.choice(b)}" for _ in range(70)})[:30]
    rows = {"directors": [(i + 1, n, rng.choice(COUNTRIES)) for i, n in enumerate(names(rng, 10))],
            "movies": [(i + 1, t, rng.randint(1, 10), rng.randint(1985, 2024), rng.choice(["Drama", "Comedy", "Action", "Sci-Fi", "Thriller"]),
                        round(rng.uniform(3.5, 9.4), 1)) for i, t in enumerate(titles)]}
    reviewers = names(rng, 12)
    rows["reviews"] = [(i + 1, rng.randint(1, len(titles)), rng.choice(reviewers), rng.randint(1, 5)) for i in range(70)]
    return ddl, rows


def sports(rng):
    ddl = """CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT, city TEXT, founded INTEGER);
CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER REFERENCES teams(id), position TEXT, age INTEGER, goals INTEGER);
CREATE TABLE matches (id INTEGER PRIMARY KEY, home_team_id INTEGER REFERENCES teams(id), away_team_id INTEGER REFERENCES teams(id), home_score INTEGER, away_score INTEGER, season INTEGER);"""
    mascots = ["Lions", "Falcons", "Wolves", "Sharks", "Bears", "Eagles", "Tigers", "Foxes"]
    cities = rng.sample(CITIES, 8)
    rows = {"teams": [(i + 1, f"{c} {m}", c, rng.randint(1890, 2010)) for i, (c, m) in enumerate(zip(cities, mascots))],
            "players": [(i + 1, n, rng.randint(1, 8), rng.choice(["Goalkeeper", "Defender", "Midfielder", "Forward"]), rng.randint(18, 38),
                         rng.randint(0, 30)) for i, n in enumerate(names(rng, 40))], "matches": []}
    for i in range(50):
        h, a = rng.sample(range(1, 9), 2)
        rows["matches"].append((i + 1, h, a, rng.randint(0, 5), rng.randint(0, 5), rng.choice([2022, 2023, 2024])))
    return ddl, rows


BUILDERS = {"shop": shop, "school": school, "library": library, "hospital": hospital, "flights": flights, "hr": hr,
            "movies": movies, "sports": sports}


def main() -> None:
    DBS.mkdir(parents=True, exist_ok=True)
    schemas = {}
    for i, (name, build) in enumerate(BUILDERS.items()):
        ddl, rows = build(random.Random(1000 + i))
        path = DBS / f"{name}.sqlite"
        path.unlink(missing_ok=True)
        con = sqlite3.connect(path)
        con.executescript(ddl)
        for table, data in rows.items():
            con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(data[0]))})", data)
        con.commit()
        con.close()
        schemas[name] = {"ddl": ddl, "held_out": name in HELD_OUT, "rows": {t: len(d) for t, d in rows.items()}}
        print(f"{name:9} {schemas[name]['rows']}{'  (held out)' if name in HELD_OUT else ''}")
    (ROOT / "data" / "schemas.json").write_text(json.dumps(schemas, indent=1))


if __name__ == "__main__":
    main()
