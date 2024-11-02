import streamlit as st
from supabase import create_client, Client

# Supabase configuration
url = "https://rsbevaaolzntbypvegcz.supabase.co"
key = "your_supabase_key"  # Replace with your actual Supabase key
supabase: Client = create_client(url, key)

# Streamlit app
st.title("Supabase Authentication")

# Function to handle email login
def email_login(email, password):
    try:
        user = supabase.auth.sign_in_with_password(email=email, password=password)
        return user
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Login with Email
st.subheader("Login with Email")
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login with Email"):
    user = email_login(email, password)
    if user:
        st.success("Logged in successfully!")
        st.write(user)

# GitHub Login
st.subheader("Login with GitHub")
if st.button("Login with GitHub"):
    # Redirect to GitHub for login
    redirect_url = f"{url}/auth/v1/authorize?provider=github"
    st.markdown(f"[Click here to log in with GitHub]({redirect_url})")

# Google Login
st.subheader("Login with Google")
if st.button("Login with Google"):
    # Redirect to Google for login
    redirect_url = f"{url}/auth/v1/authorize?provider=google"
    st.markdown(f"[Click here to log in with Google]({redirect_url})")

# Display user session
session = supabase.auth.get_session()
if session:
    st.write("Current user session:")
    st.write(session)
