
#necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#dataframe display axes
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


def loading_dataframe_country_col(filename):
    '''
    Function to load the dataframe and manipulate the country and
    rows features and return two datframes.
    '''
    df_year_test = pd.read_excel(filename,engine="openpyxl")
    df_test = pd.melt(
        df_year_test,
        id_vars=[
            'Country Name',
            'Country Code',
            'Indicator Name',
            'Indicator Code'
        ],
        var_name='Year',
        value_name='Value')
    df_country_test = df_test.pivot_table(
        index=['Year', 'Country Code', 'Indicator Name', 'Indicator Code'],
        columns='Country Name',
        values='Value').reset_index()
    df_country_test = df_country_test.drop_duplicates().reset_index()
    return df_year_test,df_country_test


def year_data(data,s_y,e_y,skip_years_freq):
    '''
    getting year wise data
    '''
    data_sub = data.copy()
    years_needed_to_analyze=[i for i in range(s_y,e_y,skip_years_freq)]
    required_col=['Country Name', 'Indicator Name']
    required_col.extend(years_needed_to_analyze)
    data_sub =  data_sub[required_col]
    data_sub = data_sub.dropna(axis=0, how="any")
    return data_sub


def filter_column_values(data,column,values):
    '''
    keep particular rows based on column values.
    '''
    data_temp= data.copy()
    data_required = \
        data_temp[data_temp[column].isin(values)].reset_index(drop=True)
    return data_required


def bar_plot_of_country(data,indicator_variable):
    df_sample = data.copy()
    df_sample.set_index('Country Name', inplace=True)
    numeric_columns_to_keep = df_sample.columns[df_sample.dtypes == 'float64']
    df_numeric_sample = df_sample[numeric_columns_to_keep]
    plt.figure(figsize=(50, 50))
    df_numeric_sample.plot(kind='bar')
    plt.title(indicator_variable)
    plt.xlabel('Country Name')
    plt.legend(title='Year', bbox_to_anchor=(1.10, 1), loc='upper right')
    plt.show()


def retrieve_indicator_data(data):
    df=data.copy()
    # Melting the DataFrame
    df_melted_s = df.melt(
        id_vars='Indicator Name',
        var_name='Year',
        value_name='Value')

    # Pivoting the DataFrame
    df_pivoted_s = df_melted_s.pivot(
        index='Year',
        columns='Indicator Name',
        values='Value')

    # Reseting index
    df_pivoted_s.reset_index(inplace=True)
    df_pivoted_s = df_pivoted_s.apply(pd.to_numeric, errors='coerce')
    del df_pivoted_s['Year']
    df_pivoted_s = df_pivoted_s.rename_axis(None, axis=1)
    return df_pivoted_s


def plott_year_wise(data,column_label):
    df = data.copy()
    df.set_index('Country Name', inplace=True)
    numeric_need = df.columns[df.dtypes == 'float64']
    df_num_need = df[numeric_need]

    plt.figure(figsize=(12, 6))
    for count in df_num_need.index:
        plt.plot(
            df_num_need.columns,
            df_num_need.loc[count],
            label=count,
            linestyle='dashed',
            marker='o')

    plt.title(column_label)
    plt.xlabel('Year')
    plt.legend(title='Country', bbox_to_anchor=(1.15, 1), loc='upper right')
    plt.show()


# Reading the Data
df_year_column, df_country_column = \
    loading_dataframe_country_col('world_bank_climate.xlsx')

#collect data from 1970 to 2020 with 5 interval timeframe
df_year_sample = year_data(df_year_column,1970,2020,5)

#10 countires for analysis regarding indicator occurance
countries_to_check = \
    df_year_sample['Country Name'].value_counts().index.tolist()[15:30]
df_year_sample['Country Name'].value_counts()

#filter dataframe for countries required
df_year_sample_country  = filter_column_values(
    df_year_sample,
    'Country Name',
    countries_to_check)

#getting indicators present wrt countries
country_dict = dict()
for i in range(df_year_sample_country.shape[0]):
    if df_year_sample_country['Country Name'][i] not in country_dict.keys():
        country_dict[
            df_year_sample_country['Country Name'][i]
        ]=[df_year_sample_country['Indicator Name'][i]]
    else:
        country_dict[df_year_sample_country['Country Name'][i]].append(
            df_year_sample_country['Indicator Name'][i])

for k,v in country_dict.items():
    country_dict[k] = set(v)

#getting common features based on respective countries.
inter = country_dict['Belgium']
for v in country_dict.values():
    inter = inter.intersection(v)
print(df_year_sample.describe())

df_year_sam_co2 = filter_column_values(
    df_year_sample,'Indicator Name',
    ['CO2 emissions from solid fuel consumption (kt)'])
print(df_year_sam_co2.describe())

df_year_sam_co2_cont  = filter_column_values(
    df_year_sam_co2,
    'Country Name',
    countries_to_check)
bar_plot_of_country(df_year_sam_co2_cont,'CO2 emissions from solid fuel consumption (kt)')

df_year_col_mort = filter_column_values(
    df_year_sample,
    'Indicator Name',
    [
     "Electricity production from renewable sources, excluding hydroelectric (% of total)"
    ])
df_year_col_mort = filter_column_values(
    df_year_col_mort,
    'Country Name',
    countries_to_check)
print(df_year_col_mort.describe())
bar_plot_of_country(
    df_year_col_mort,
    'Electricity production from renewable sources, excluding hydroelectric (% of total)')


df_year_high_income= filter_column_values(
    df_year_sample,
    'Country Name',
    ['High income'])
data_heat_map_high_income = retrieve_indicator_data(df_year_high_income)


features_to_check = [
    'Agricultural land (sq. km)',
    'CO2 emissions from gaseous fuel consumption (kt)',
    'Electricity production from coal sources (% of total)',
    'Electricity production from renewable sources, excluding hydroelectric (% of total)',
    'Energy use (kg of oil equivalent per capita)',
    'Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)',
    'Urban population growth (annual %)']

data_heat_map_high_income_map = data_heat_map_high_income[features_to_check]
data_heat_map_high_income_map.corr()
sns.heatmap(
    data_heat_map_high_income_map.corr(),
    annot=True, cmap='YlGnBu', linewidths=.7, fmt='.3g')

df_year_col_energy= filter_column_values(
    df_year_sample,
    'Indicator Name',
    ['Electricity production from coal sources (% of total)'])
df_year_col_energy  = filter_column_values(
    df_year_col_energy,
    'Country Name',
    countries_to_check)
print(df_year_col_energy.describe())

df_year_col_urban= filter_column_values(
    df_year_sample,
    'Indicator Name',
    ['Urban population (% of total population)'])
df_year_col_urban  = filter_column_values(
    df_year_col_urban, 'Country Name', countries_to_check)
plott_year_wise(
    df_year_col_energy,
    'Electricity production from coal sources (% of total)')

df_year_co2= filter_column_values(
    df_year_sample,
    'Indicator Name',
    ['CO2 emissions from gaseous fuel consumption (kt)'])
df_year_co2  = filter_column_values(
    df_year_co2,
    'Country Name',
    countries_to_check)
print(df_year_co2.describe())
plott_year_wise(
    df_year_co2,
    'CO2 emissions from gaseous fuel consumption (kt)')

df_year_oecd = filter_column_values(
    df_year_sample,
    'Country Name',
    ['OECD members'])
data_heat_map_oecd = retrieve_indicator_data(df_year_oecd)
data_heat_map_oecd_s = data_heat_map_oecd[features_to_check]
sns.heatmap(data_heat_map_oecd_s.corr(),
            annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')

df_year_col_post= filter_column_values(
    df_year_sample, 'Country Name', ['Belgium'])
data_heat_map_post = retrieve_indicator_data(df_year_col_post)
data_heat_map_post_sub = data_heat_map_post[features_to_check]
plt.figure()
sns.heatmap(
    data_heat_map_post_sub.corr(),
    annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')




