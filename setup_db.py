import sqlite3, os, random
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/support.db")
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS customers (
  id INTEGER PRIMARY KEY,
  name TEXT, email TEXT, plan TEXT, joined_date TEXT)""")

cur.execute("""CREATE TABLE IF NOT EXISTS tickets (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER, subject TEXT,
  status TEXT, created_at TEXT,
  FOREIGN KEY(customer_id) REFERENCES customers(id))""")

# Idempotency guard — only seed if tables are empty
cur.execute("SELECT COUNT(*) FROM customers")
if cur.fetchone()[0] > 0:
    print("Database already seeded. Skipping.")
    conn.close()
    exit(0)

names = ["Alice Chen","Bob Smith","Carol White","David Lee",
  "Emma Brown","Frank Kim","Grace Liu","Henry Park",
  "Iris Patel","James Wilson","Karen Davis","Leo Martinez",
  "Mia Johnson","Noah Taylor","Olivia Wang","Peter Hall",
  "Quinn Adams","Rachel Green","Sam Torres","Tina Nguyen"]

plans = ["free","starter","pro","enterprise"]
subjects = [
  "Can't log in to my account",
  "Charge on my card I don't recognize",
  "How do I cancel my subscription?",
  "Request a refund for last month",
  "App keeps crashing on mobile",
  "Need to update billing info",
  "Transfer my data to another account",
  "Promo code not working",
  "Upgrade my plan",
  "Download my invoice"]
statuses = ["open","in_progress","resolved","closed"]

for i, name in enumerate(names, 1):
    email = name.lower().replace(" ", ".") + "@email.com"
    plan = random.choice(plans)
    joined = (datetime.now() - timedelta(days=random.randint(10, 730))).date()
    cur.execute("INSERT INTO customers VALUES (?,?,?,?,?)",
                (i, name, email, plan, str(joined)))

for i in range(1, 21):
    cid = random.randint(1, 20)
    subj = random.choice(subjects)
    status = random.choice(statuses)
    created = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
    cur.execute("INSERT INTO tickets VALUES (?,?,?,?,?)",
                (i, cid, subj, status, created))

conn.commit()
conn.close()
print("Done! Database created at data/support.db")
