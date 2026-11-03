
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

# Read all seven SQL queries
with open("sql/queries.sql", "r", encoding="utf-8") as file:
    sql_text = file.read()

# Separate queries using semicolons
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

print("TOTAL SQL QUERIES:", len(queries))

# Execute each query
for i, query in enumerate(queries, start=1):

    print("\n" + "=" * 50)
    print(f"QUERY {i} RESULTS")
    print("=" * 50)

    result = pd.read_sql_query(query, conn)

    print(result.head(10))
    print("\nTotal rows:", len(result))

conn.close()

print("\nALL SQL QUERIES EXECUTED SUCCESSFULLY!")
