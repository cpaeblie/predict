import streamlit as st
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = "https://rsbevaaolzntbypvegcz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJzYmV2YWFvbHpudGJ5cHZlZ2N6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzA0MzUxODEsImV4cCI6MjA0NjAxMTE4MX0.wSsAMedFR5nJkSIbeYL3g_lFx_99Z9GtX383VC5nGus"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to handle email login
def email_login(email, password):
    response = supabase.auth.sign_in_with_password(email=email, password=password)
    return response

# Function to handle GitHub login
def github_login():
    return supabase.auth.sign_in_with_provider('github')

# Function to handle Google login
def google_login():
    return supabase.auth.sign_in_with_provider('google')

# Main app function
def main():
    st.title("SmartExam Creator - Login")

    # Login form
    login_option = st.selectbox("Select Login Method", ["Email", "GitHub", "Google"])

    if login_option == "Email":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            response = email_login(email, password)
            if response.user:
                st.success(f"Welcome {response.user.email}!")
                # Proceed to main application logic
            else:
                st.error("Login failed! Please check your credentials.")

    elif login_option == "GitHub":
        if st.button("Login with GitHub"):
            response = github_login()
            st.success("Redirecting to GitHub for authentication...")
            # The actual redirection will be handled by Supabase

    elif login_option == "Google":
        if st.button("Login with Google"):
            response = google_login()
            st.success("Redirecting to Google for authentication...")
            # The actual redirection will be handled by Supabase

if __name__ == '__main__':
    main()
