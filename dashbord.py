import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Telecom Data user and customer Analyis Dashboard",
    page_icon=":star:",  # You can use an emoji or provide a path to an icon file
    layout="wide"  # Adjust layout to use more screen width
)

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color:  hsl(130, 35%, 54%);
            text-align: center;
            margin-top: 20px;
        }
        .text {
            font-size: 20px;
            color: hsl(129, 27%, 71%);
            text-align: center;
            margin-top: 20px;
            padding: 0 20px;
        }
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Layout container
with st.container():
    # Title
    st.markdown('<div class="title">Telecom Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">User And Customer Dashboard</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="text">Welcome to the dashboard. Here you can view the details of the most satisfied customers based on recent analyses.</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    
st.write("<br>", unsafe_allow_html=True)
