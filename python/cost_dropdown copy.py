import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

data = [
    {
        "Organizer": "ACCORD Tribals",
        "Insurer": "ACCORD Royal Sundaram Insurance Company",
        "Administrator": "ACCORD",
        "Provider": "ACCORD hospital",
        "Premium": "Rs. 30 per person per year",
        "Benefit Package": "Hospitalization expenses up to a maximum limit of Rs. 3000. No exclusions",
        "Risk Management": "Collection period, salary for providers, essential medicines and STGs"
    },
    {
        "Organizer": "Karuna Trust",
        "Insurer": "National Insurance Company",
        "Administrator": "Karuna Trust",
        "Provider": "Government hospitals",
        "Premium": "Rs. 20 per person per year",
        "Benefit Package": "Medicine cost @ Rs. 50 per inpatient day. Loss of wages @ Rs. 50 per inpatient day",
        "Risk Management": "Collection period, flat rate"
    },
    {
        "Organizer": "Yeshasvini Members of the cooperative societies",
        "Insurer": "Yeshasvini Trust",
        "Administrator": "Yeshasvini Trust",
        "Provider": "Family Health Plan Ltd",
        "Premium": "Rs. 120 per person per year",
        "Benefit Package": "Cover for surgeries up to a maximum of Rs. 200,000 per patient per year",
        "Risk Management": "Collection period, only surgical conditions, pre-authorization, tariffs fixed for procedures, photo id card"
    },
    {
        "Organizer": "RAHA Tribals",
        "Insurer": "RAHA",
        "Administrator": "RAHA",
        "Provider": "RAHA Network of 'mission' clinics and hospitals",
        "Premium": "Rs. 20 per person per year",
        "Benefit Package": "Unlimited OP cover, hospitalization cover for a maximum of Rs. 1250",
        "Risk Management": "Collection period, salary for providers, strict referral system, copayments"
    },
    {
        "Organizer": "JRHIS Farmers",
        "Insurer": "JRHIS",
        "Administrator": "JRHIS",
        "Provider": "JRHIS MG medical college",
        "Premium": "Rs. 100 per family per year",
        "Benefit Package": "OP cover by VHWs, hospital cover at medical college",
        "Risk Management": "Family as the enrolment unit, collection period, referral system"
    },
    {
        "Organizer": "SEWA Self-employed women and their dependents",
        "Insurer": "ICICI–Lombard",
        "Administrator": "SEWA",
        "Provider": "Public and private hospitals",
        "Premium": "Rs. 85 per person per year",
        "Benefit Package": "Hospitalization expenses up to Rs. 2000 per patient per year",
        "Risk Management": "Collection period"
    },
    {
        "Organizer": "Student’s Health Home",
        "Insurer": "SHH",
        "Administrator": "SHH",
        "Provider": "SHH",
        "Premium": "Rs. 5 per student per year",
        "Benefit Package": "Unlimited OP and IP at SHH-run facilities",
        "Risk Management": "School is the enrolment unit, providers paid fixed salaries. Definite collection period, referral system, mandatory cover of the entire population"
    },
    {
        "Organizer": "VHS Rural population",
        "Insurer": "VHS",
        "Administrator": "VHS",
        "Provider": "VHS",
        "Premium": "Rs. 100 per person per year",
        "Benefit Package": "Hospitalization expenses up to maximum limits",
        "Risk Management": "Nil"
    },
    {
        "Organizer": "Universal Health Insurance Scheme",
        "Insurer": "4 public sector insurance companies",
        "Administrator": "? Any hospital",
        "Provider": "?",
        "Premium": "Rs. 548 for a family of five (Rs. 300 subsidized by the government)",
        "Benefit Package": "Hospitalization cover up to a maximum limit of Rs. 30,000 per family per year. Personal accident up to Rs. 25,000. Loss of wages @ Rs. 50 per patient day",
        "Risk Management": "Family as the enrolment unit, waiting period"
    },
    {
        "Organizer": "Kudumbashree (proposed)",
        "Insurer": "Kudumbashree and Govt of Kerala",
        "Administrator": "ICICI-Lombard",
        "Provider": "SHGs Empanelled hospitals",
        "Premium": "Rs. 399 per family per year, Rs. 366 subsidized by the government",
        "Benefit Package": "Hospitalization up to a maximum of Rs. 30,000 per family per year. No exclusions. Personal accident up to Rs. 100,000. Loss of wages @ Rs. 50 per patient day for a week",
        "Risk Management": "Family as the enrolment unit"
    },
    {
        "Organizer": "AP Scheme (proposed)",
        "Insurer": "4 public sector insurance companies",
        "Administrator": "A TPA",
        "Provider": "Empanelled hospitals",
        "Premium": "Rs. 548 for a family of five (Rs. 400 subsidized by the government)",
        "Benefit Package": "Hospitalization expenses up to Rs. 25,000 for surgical conditions and Rs. 75,000 for serious conditions. But, only for the first 3 days for medical conditions",
        "Risk Management": "Waiting period. Copayments after 3 days for medical conditions"
    },
    {
        "Organizer": "Karnataka Scheme (proposed)",
        "Insurer": "Karnataka Government and 4 public sector companies",
        "Administrator": "Department of health staff (for collection of premium)",
        "Provider": "Any hospitals, especially public sector hospitals",
        "Premium": "Rs. 548 for a family of five. Rs. 300 subsidy from the government",
        "Benefit Package": "Hospitalization cover up to a maximum limit of Rs. 30,000 per family per year. Personal accident up to Rs. 25,000. Loss of wages @ Rs. 50 per patient day",
        "Risk Management": "Waiting period, family as the enrolment unit"
    },
    {
        "Organizer": "Assam Scheme",
        "Insurer": "Assam Government",
        "Administrator": "ICICI-Lombard",
        "Provider": "Hospitalization expenses up to a maximum of Rs. 25,000 for select disease conditions, e.g. cancer, IHD, renal failure, stroke etc",
        "Premium": "All Assam citizens except government servants/those with more than Rs. 2 lakh per annum income",
        "Benefit Package": "Hospitalization expenses up to a maximum of Rs. 25,000 for select disease conditions, e.g. cancer, IHD, renal failure, stroke etc",
        "Risk Management": "Mandatory cover of the entire population"
    }
]

# Creating DataFrame
df_full_schemes = pd.DataFrame(data)

# Assuming df_full_schemes is the DataFrame you already have

# Create a dropdown widget with organizers as options
dropdown = widgets.Dropdown(
    options=df_full_schemes['Organizer'].tolist(),
    value=df_full_schemes['Organizer'].tolist()[0],
    description='Organizer:',
    disabled=False,
)

# Function to handle dropdown change event
def dropdown_eventhandler(change):
    clear_output(wait=True)
    display(dropdown)  # To display the dropdown

    # Filter the DataFrame based on the selected organizer
    filtered_data = df_full_schemes[df_full_schemes['Organizer'] == change.new]

    # Display the details of the schemes related to the selected organizer
    display(filtered_data)

# Display the dropdown
display(dropdown)

# Bind the event handler to the dropdown
dropdown.observe(dropdown_eventhandler, names='value')
