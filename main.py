import streamlit as st

st.markdown("# Main page 🎈")
st.markdown("""
这个是一个CSI信号的可视化工具，可以用来分析CSI信号的特征。
""")
st.markdown("""
由于在开发时没有考虑多人同时使用的情况，所以可能会出现一些问题。为了避免这些问题，建议在使用时只有一个人使用。
同时，应用会在整点和半点的时候自动重启，以排除由于多人同时使用导致的问题。
""")