import streamlit as st
from supabase import create_client, Client

# Inisialisasi Supabase
url = "https://rsbevaaolzntbypvegcz.supabase.co"  # Ganti dengan URL Supabase Anda
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJzYmV2YWFvbHpudGJ5cHZlZ2N6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzA0MzUxODEsImV4cCI6MjA0NjAxMTE4MX0.wSsAMedFR5nJkSIbeYL3g_lFx_99Z9GtX383VC5wGus"  # Ganti dengan Anon Key dari Supabase Anda
supabase: Client = create_client(url, key)

# Fungsi Sign-up
def signup(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        st.error("Sign-up gagal: " + str(e))
        return None

# Fungsi Login
def login(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user
    except Exception as e:
        st.error("Login gagal: " + str(e))
        return None

# Fungsi Logout
def logout():
    supabase.auth.sign_out()
    st.success("Berhasil logout")

# UI Streamlit
st.title("Aplikasi Sign-up dan Login")

menu = st.sidebar.selectbox("Menu", ["Sign Up", "Login", "Logout"])

if menu == "Sign Up":
    st.subheader("Buat Akun Baru")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        user = signup(email, password)
        if user:
            st.success("Sign-up berhasil! Silakan login.")

elif menu == "Login":
    st.subheader("Masuk ke Akun")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login(email, password)
        if user:
            st.success("Login berhasil!")

elif menu == "Logout":
    st.subheader("Logout")
    if st.button("Logout"):
        logout()
