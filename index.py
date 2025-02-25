import streamlit as st
def main():
    # Create a horizontal navigation bar
    st.title("My Streamlit App")
    
    # Using st.tabs for a horizontal navigation
    tab1, tab2, tab3 = st.tabs(["Home", "Data", "About"])
    
    with tab1:
        st.header("Home")
        st.write("Welcome to the home page!")
    
    with tab2:
        st.header("Data")
        st.write("This is where you would display your data.")
    
    with tab3:
        st.header("About")
        st.write("Information about this application.")

if __name__ == "__main__":
    main()