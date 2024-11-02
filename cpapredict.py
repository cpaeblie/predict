import streamlit as st
from supabase import create_client, Client

# Supabase configuration
url = "https://rsbevaaolzntbypvegcz.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJzYmV2YWFvbHpudGJ5cHZlZ2N6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzA0MzUxODEsImV4cCI6MjA0NjAxMTE4MX0.wSsAMedFR5nJkSIbeYL3g_lFx_99Z9GtX383VC5wGus"
supabase: Client = create_client(url, key)

# Streamlit app
st.title("Supabase Authentication")

# Login with Email
st.subheader("Login with Email")
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login with Email"):
    try:
        user = supabase.auth.sign_in_with_password(email=email, password=password)
        st.success("Logged in successfully!")
        st.write(user)
    except Exception as e:
        st.error(f"Error: {e}")

# Login with GitHub
st.subheader("Login with GitHub")
if st.button("Login with GitHub"):
    try:
        redirect_url = supabase.auth.sign_in_with_github()
        st.write("Redirecting to GitHub for login...")
        st.markdown(f"[Click here if not redirected]({redirect_url})")
    except Exception as e:
        st.error(f"Error: {e}")

# Login with Google
st.subheader("Login with Google")
if st.button("Login with Google"):
    try:
        redirect_url = supabase.auth.sign_in_with_google()
        st.write("Redirecting to Google for login...")
        st.markdown(f"[Click here if not redirected]({redirect_url})")
    except Exception as e:
        st.error(f"Error: {e}")

# Display user session
session = supabase.auth.get_session()
if session:
    st.write("Current user session:")
    st.write(session)
