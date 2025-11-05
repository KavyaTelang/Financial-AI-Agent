# streamlit_app.py (AGGRESSIVE INSTALLER DIAGNOSTIC)

import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Installation Diagnostics", layout="wide")
st.title("🔬 Aggressive Dependency Installation")

st.info("This script will attempt to install the required packages directly.")

# The list of packages we need to install
packages = [
    "phidata==2.7.7",
    "python-dotenv",
    "duckduckgo-search",
    "groq",
    "openai",
    "alpha-vantage==3.0.0"
]

# --- Installation Phase ---
st.header("1. Attempting to Install Packages")

for package in packages:
    st.write(f"--- Installing `{package}` ---")
    try:
        # We use subprocess to run the pip install command
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True,
            text=True,
            check=True  # This will raise an error if pip fails
        )
        st.success(f"Successfully installed `{package}`.")
        # Show the output from pip
        st.code(result.stdout)
    except subprocess.CalledProcessError as e:
        # This will catch errors from the pip command itself
        st.error(f"FAILED to install `{package}`. Pip returned a non-zero exit code.")
        st.write("Pip Standard Output:")
        st.code(e.stdout)
        st.write("Pip Standard Error:")
        st.code(e.stderr)
    except Exception as e:
        # This will catch other errors, like the command not being found
        st.error(f"An unexpected error occurred while trying to install `{package}`: {e}")

# --- Verification Phase ---
st.header("2. Verifying Installation (`pip list`)")

try:
    # Run pip list again to see if the packages are there now
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    st.code(result.stdout)
except Exception as e:
    st.error(f"Could not run 'pip list' after installation attempts: {e}")


# --- Import Test Phase ---
st.header("3. Attempting to Import `alpha_vantage.newsandsentiment`")
try:
    st.info("Now attempting: from alpha_vantage.newsandsentiment import NewsAndSentiment")
    from alpha_vantage.newsandsentiment import NewsAndSentiment
    st.success("✅ SUCCESS: The module was imported correctly after manual installation!")
except Exception as e:
    st.error("❌ FAILED: The import still failed. See the package installation logs above for clues.")
    st.code(str(e))