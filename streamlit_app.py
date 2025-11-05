# streamlit_app.py (DIAGNOSTIC MODE)

import streamlit as st
import sys
import subprocess
import traceback

st.set_page_config(page_title="System Diagnostics", layout="wide")
st.title("🔬 System Environment Diagnostics")

st.header("1. Python Version")
st.code(sys.version)

st.header("2. Installed Packages (`pip list`)")
try:
    # Run the 'pip list' command to see all installed packages and their versions
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    st.code(result.stdout)
    if result.stderr:
        st.error("Errors from pip list:")
        st.code(result.stderr)
except Exception as e:
    st.error(f"Could not run 'pip list': {e}")

st.header("3. Attempting to Import `alpha_vantage.newsandsentiment`")
try:
    st.info("Attempting: from alpha_vantage.newsandsentiment import NewsAndSentiment")
    
    # We try to import the module
    from alpha_vantage.newsandsentiment import NewsAndSentiment
    
    st.success("✅ SUCCESS: The module was imported correctly!")
    
    # Let's see where the library is installed
    import alpha_vantage
    st.write("Location of the `alpha_vantage` library:")
    st.code(alpha_vantage.__file__)
    
except ModuleNotFoundError:
    st.error("❌ FAILED: ModuleNotFoundError was raised.")
    st.write("This confirms the module does not exist at that path in the installed library.")
    st.write(traceback.format_exc())
except Exception as e:
    st.error(f"❌ FAILED with a different error: {type(e).__name__}")
    st.write(traceback.format_exc())