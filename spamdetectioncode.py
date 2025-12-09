import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt

# ------------------------------------
# CREATE SPARK SESSION
# ------------------------------------
spark = SparkSession.builder \
    .appName("SpamDetectionApp") \
    .master("local[*]") \
    .getOrCreate()

# ------------------------------------
# LOAD CSV USING SPARK
# ------------------------------------
df = spark.read.csv("spam_dataset.csv", header=True)
df.count()

# ------------------------------------
# SPAM LOGIC UDF
# ------------------------------------
spam_words = ["free", "win", "won", "claim", "click", "offer", "jackpot", "bonus", "vacation", "gift"]

def label_text(text):
    if text is None:
        return "NOT SPAM"

    text = text.lower()
    score = sum(word in text for word in spam_words)

    if score >= 3:
        return "SPAM"
    elif score == 2:
        return "LESS SPAM"
    else:
        return "NOT SPAM"

label_udf = udf(label_text, StringType())

df = df.withColumn("status", label_udf(col("text")))

# ------------------------------------
# CONVERT TO PANDAS FOR DISPLAY
# ------------------------------------
pdf = df.toPandas()

# ------------------------------------
# COLOR MAPPING FOR STREAMLIT
# ------------------------------------
def color_status(val):
    if val == "SPAM":
        return "background-color: red; color: white"
    elif val == "LESS SPAM":
        return "background-color: yellow; color: black"
    else:
        return "background-color: green; color: white"

# ------------------------------------
# STREAMLIT UI
# ------------------------------------
st.title("🔥 Spam Detection App (Spark Enabled)")

st.subheader("📌 Dataset Preview with Spark Labeling")
st.dataframe(pdf.style.applymap(color_status, subset=["status"]))

# ------------------------------------
# USER MESSAGE CHECKER
# ------------------------------------
st.subheader("🔍 Check Message Using Spark Logic")

user_msg = st.text_input("Enter your message:")

if st.button("Predict"):
    result = label_text(user_msg)

    if result == "SPAM":
        st.error("🚨 SPAM DETECTED!")
    elif result == "LESS SPAM":
        st.warning("⚠️ LESS SPAM POSSIBLE")
    else:
        st.success("✔️ NOT SPAM")

# ------------------------------------
# CHART
# ------------------------------------
st.subheader("📊 Spam Statistics")

counts = pdf["status"].value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values, color=["green", "yellow", "red"])
ax.set_title("Spam Category Distribution")
ax.set_ylabel("Count")

st.pyplot(fig)
