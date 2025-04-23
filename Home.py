import streamlit as st

# Centered title & text via HTML/CSS in Markdown
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Trading App Dashboard</h1>
        <p>Welcome to the Trading App. Use the sidebar to navigate between the Best_Performance_Data and Realitime&backtest pages.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Center the GIF using columns
gif_path = r"FYP\dashboard\angry-2498.gif"
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(gif_path, use_container_width=True)




# & "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run "C:\Users\user\OneDrive - Asia Pacific University\Data\FYP\my_appFYP\home.py"
# streamlit run "C:\Users\xuank\OneDrive - Asia Pacific University\Data\FYP\my_appFYP\home.py"