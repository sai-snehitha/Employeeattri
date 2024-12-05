import streamlit as st
import psycopg2
import pandas as pd
from psycopg2 import OperationalError

# Database connection setup
def connect_to_database():
    try:
        conn = psycopg2.connect(
            host="localhost",  # our PostgreSQL host
            database="1TestEmployee",  # our database name
            user="postgres",  # our PostgreSQL username
            password="Snehitha@23",  # our PostgreSQL password
            port="5432" # our website port
        )
        return conn
    except OperationalError as e:
        st.error(f"Cannot connect to the database: {e}")
        return None

# Execute query function
def execute_query(query):
    conn = connect_to_database()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if cursor.description:  # If the query returns data
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=columns)
                else:  # For queries like INSERT, UPDATE, DELETE
                    conn.commit()
                    return "Query executed successfully."
        except Exception as e:
            return f"Execution query error: {e}"
        finally:
            conn.close()
    else:
        return "Connection failed to the database."

# Sidebar
st.sidebar.title("Employee Attrition")
st.sidebar.markdown("Navigate through the options below:")

options = st.sidebar.selectbox(
    "Select an Option",
    [
        "Home",
        "Query Attrition Details",
        "Query Commute and Absenteeism",
        "Employee Details",
        "Performance Analysis"
    ]
)

# 1. Home
if options == "Home":
    st.title("Welcome to the Employee Attrition Analysis")
    st.markdown("""
    This application provides insights into employee attrition data and allows users to query, visualize, and analyze the data interactively. Use the sidebar to navigate through different features.
    """)

    # Test database connection
    st.markdown("Testing database connection...")
    conn = connect_to_database()
    if conn:
        st.success("Successfully connected to the PostgreSQL database!")
        conn.close()
    else:
        st.error("Failed to connect to the database. Check connection settings.")

# 2. Query Attrition Details
elif options == "Query Attrition Details":
    st.title("Attrition Details")
    st.markdown("Query employee attrition data by job role or department.")

    # User input for job role
    jobrole_id = st.number_input("Enter Job Role ID", min_value=1, step=1)
    query = f"""
    SELECT 
        a.jobroleid, 
        j.department, 
        j.job_role, 
        a.reason_for_leaving, 
        a.company_culture_fit, 
        a.team_dynamics, 
        a.company_loyalty
    FROM attrition a
    JOIN jobrole j ON a.jobroleid = j.jobroleid
    WHERE a.jobroleid = {jobrole_id};
    """

    if st.button("Execute Query"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        elif isinstance(result, pd.DataFrame) and result.empty:
            st.warning("No records found for the given Job Role ID.")
        else:
            st.error(result)

# 3. Query Commute and Absenteeism
elif options == "Query Commute and Absenteeism":
    st.title("Commute and Absenteeism Data")
    st.markdown("View commute distance and absenteeism rates for employees.")

    query = """
    SELECT 
        c.employee_id, 
        e.age, 
        e.gender, 
        c.commute_distance, 
        c.absenteeism_rate
    FROM commuteandabsenteeism c
    JOIN employee e ON c.employee_id = e.employee_id;
    """

    if st.button("Fetch Commute Data"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        else:
            st.warning("No commute and absenteeism data found.")

# 4. Employee Details
elif options == "Employee Details":
    st.title("Search Employee Details")
    st.markdown("Search for an employee by their ID and view all available details.")

    # Input for employee search
    emp_id = st.text_input("Enter Employee ID")

    query = f"""
    SELECT 
        e.employee_id, 
        e.age, 
        e.gender, 
        e.marital_status, 
        j.department, 
        j.job_role, 
        j.salary
    FROM employee e
    JOIN jobrole j ON e.jobroleid = j.jobroleid
    WHERE e.employee_id = '{emp_id}';
    """

    if st.button("Search Employee"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        else:
            st.warning("No employee details found for the given Employee ID.")

# 5. Performance Analysis
elif options == "Performance Analysis":
    st.title("Performance Analysis")
    st.markdown("Analyze employee performance and work-life balance.")

    query = """
    SELECT 
        p.employee_id, 
        e.age, 
        e.gender, 
        p.performance_rating, 
        p.work_life_balance_rating
    FROM performance p
    JOIN employee e ON p.employee_id = e.employee_id;
    """

    if st.button("Analyze Performance"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        else:
            st.warning("No performance data available.")
