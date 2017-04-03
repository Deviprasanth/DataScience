import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


def value_entry(a,b):
    value=b[len(b)-1]
    for i,val in enumerate(b):
        if a<b[i]:
            value = b[i-1]
            return value
    return value

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    towndf = pd.read_csv('university_towns.txt', sep='\n',header=None)
    towndf.columns=['Entry']
    length = len(towndf)
    stateindex_list = towndf[towndf['Entry'].str.contains(u'\[edit\]')].index.tolist()
    
    towndf['State']=towndf.apply(lambda row:towndf['Entry'].iloc[value_entry(row.name,stateindex_list)],axis=1).str.replace(r'\[.*', '')
    towndf= towndf.drop(towndf.index[stateindex_list])
    towndf['Entry']=towndf['Entry'].str.replace(r'\ \(.*', '')
    towndf.rename(columns={'Entry':'RegionName'},inplace = True)
    return towndf

def recstart(a):
    val=len(a)-1
    for i in range(len(a)-3):
        if (a[i]>a[i+1] and a[i+1]>a[i+2] ):
            return i
    return val

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    GDP_Q=pd.read_excel('gdplev.xls',sheetname='Sheet1',skiprows=5).iloc[214:280,4:-1].rename(columns={'Unnamed: 4':'Quarter'})
    GDP_Q = GDP_Q.reset_index().drop('index',axis=1)
    
    #return GDP_Q
    return GDP_Q.iloc[recstart(GDP_Q['GDP in billions of current dollars.1'])]['Quarter']

def recend(rs,a):
    val=len(a)-1-rs
    for i in range(val-2):
        if (a[rs+i]<a[rs+i+1] and a[rs+i+1]<a[rs+i+2]):
            #print(i)
            return i+2
    return val

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    GDP_Q=pd.read_excel('gdplev.xls',sheetname='Sheet1',skiprows=5).iloc[214:280,4:-1].rename(columns={'Unnamed: 4':'Quarter'})
    GDP_Q = GDP_Q.reset_index().drop('index',axis=1)
    
    #return GDP_Q
    rstrt = recstart(GDP_Q['GDP in billions of current dollars.1'])
    #print(rstrt,recend(rstrt,GDP_Q['GDP in billions of current dollars.1']))
    return GDP_Q.iloc[rstrt+recend(rstrt,GDP_Q['GDP in billions of current dollars.1'])]['Quarter']

def recbottom(rs,re,a):
    val=re-2
    for i in range(re-rs):
        return a[rs+i].argmin()
    return val

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    GDP_Q=pd.read_excel('gdplev.xls',sheetname='Sheet1',skiprows=5).iloc[214:280,4:-1].rename(columns={'Unnamed: 4':'Quarter'})
    GDP_Q = GDP_Q.reset_index().drop('index',axis=1)
    rstrt = recstart(GDP_Q['GDP in billions of current dollars.1'])
    rend = recend(rstrt,GDP_Q['GDP in billions of current dollars.1'])
    #return GDP_Q
    return GDP_Q.iloc[rstrt + recbottom(rstrt,rend,GDP_Q['GDP in billions of current dollars.1'])]['Quarter']

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    homedf = pd.read_csv('City_Zhvi_AllHomes.csv')
    homedf =homedf.replace({'State':states})
    homedf = homedf.set_index(["State","RegionName"]).iloc[:,49:]
    
    return homedf.groupby(pd.PeriodIndex(homedf.columns, freq='q'), axis=1).mean()

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    university_towns = get_list_of_university_towns()
    homedf = convert_housing_data_to_quarters()
    
    ratio_df = pd.DataFrame({'ratio': homedf[get_recession_start()]/homedf[get_recession_bottom()]})
    ratio_df = ratio_df.set_index(homedf.index)
    
    university_towns = university_towns.set_index(['State', 'RegionName'])
    
    ratio_college = ratio_df.loc[list(university_towns.index)]['ratio'].dropna()
    
    ratio_not_college_indices = set(ratio_df.index) - set(ratio_college.index)
    
    ratio_not_college = ratio_df.loc[list(ratio_not_college_indices),:]['ratio'].dropna()
        
    statistic, p_value = tuple(ttest_ind(ratio_college, ratio_not_college))
    
    outcome = statistic < 0
    
    different = p_value < 0.05
    
    better = ["non-university town", "university town"]
    
    return (different, p_value, better[outcome])

#homedf[get_recession_start()].div(homedf[get_recession_bottom()])

print(run_ttest())


