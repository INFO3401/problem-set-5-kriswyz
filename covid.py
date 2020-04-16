# Utility functions for epidemiology data
import pandas as pd

# Use this to change any date information to the right type
# and to move all date data from columns to rows
def correctDateFormat(df):
    # Move date data from columns to rows. Will create two new columns (one for
    # date and one for the number of confirmed cases). Will add a new row for
    # each date x province/state
    df = df.melt(id_vars=df.columns[0:4], var_name="Date", value_name="Confirmed")

    # Convert date to a datetime object so pandas knows how to do math with it
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Helper function you can use to group the data for a given country of interestself.
# Just pass the function your dataframe and the country's name as a string
def groupByCountry(df, country):
    data = df.loc[df["Country/Region"] == country]
    return data.groupby("Date", as_index=False).sum()

#4:
def correctDateFormat(df):
	df = df.melt(id_vars=df.columns[0:4],var_name="Date",value_name="Confirmed")

#6:
	df["Data"] = pd.to_datetime(df["Date"])
	return df

#14:
def aggregateCountry(df,country):
	CountryData = df.iloc[df["Country/Region"] == country]
	return CountryData.groupby("Date", index = False).sum()