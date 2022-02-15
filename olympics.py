import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from clean_file import clean

data = pd.read_csv('results_fixed.csv')
df = clean(data)
seen = {}

FIELD = [
    'Decathlon Men',
    'Heptathlon Women',
    'Discus Throw Men',
    'Discus Throw Women',
    'Hammer Throw Men',
    'Hammer Throw Women',
    'High Jump Men',
    'High Jump Women',
    'Javelin Throw Men',
    'Javelin Throw Women',
    'Long Jump Men',
    'Long Jump Women',
    'Pole Vault Men',
    'Pole Vault Women',
    'Shot Put Men',
    'Shot Put Women',
    'Triple Jump Men',
    'Triple Jump Women',
    ]

def main():
    performances_over_time().show()
    gold_medal_map().show()
    simulated_games = simulate()
    print('Simulated Olympic Results')
    print(simulated_games)

    # save to csv
    simulated_games.to_csv('output.csv')

def convert_to_seconds(mark):
    """
    Takes in a performance mark as a string, converts it to seconds, and
    returns it as a float
    """

    if mark.count(':') == 1:
        t = datetime.strptime(mark, '%M:%S.%f')
        return t.minute * 60 + t.second + t.microsecond / 1000000
    elif mark.count(':') == 2:

        if mark.count('.') == 1:
            t = datetime.strptime(mark, '%H:%M:%S.%f')
        else:
            t = datetime.strptime(mark, '%H:%M:%S')
        return (t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1000000
    else:
        return float(mark)


def convert_from_seconds(mark):
    """
    Takes in a performance mark as a float, converts it from seconds to
    hours/minutes/seconds and returns it as a string
    """
    import datetime  # need to import here because global variable uses
    # datetime from datetime

    time = str(datetime.timedelta(seconds=mark))

    if time[0] == '0' and time[2] == '0' and time[3] == '0' and time[5] == '0':
        return time[6:].strip('0')
    if time[0] == '0' and time[2] == '0' and time[3] == '0' and time[5] != '0':
        return time[5:].strip('0')
    elif time[0] == '0' and time[2] == '0' and time[3] != '0':
        return time[3:].strip('0')
    elif time[0] == '0' and time[2] != '0':
        return time[2:].strip('0')
    else:
        return time.strip('0')

def get_scores(result, df):
    """
    Takes in a single result, calculates its IAAF score based on its event,
    and returns it as a float
    Made to be used on entire dataset and not individual mark
    """

    if result not in seen:
        constants = df.loc[df['Result'] == result, 'Constants'].iloc[0]
        seen[result] = 1
    else:
        i = seen[result]
        constants = df.loc[df['Result'] == result, 'Constants'].iloc[i]
        seen[result] += 1

    result_shift = constants[0]
    conversion_factor = constants[1]
    point_shift = constants[2]

    return round(conversion_factor * (result + result_shift) ** 2 + point_shift)

def get_constants(event):
    """
    Takes in an event as a string parameter, puts its result shift,
    conversion factor, and point shift into a list based on the event,
    and returns the list
    """

    # returns list [result shift, conversion factor, point shift]

    # Men's running events
    if event == '100M Men':
        return [-17, 24.63, 0]
    if event == '200M Men':
        return [-35.5, 5.08, 0]
    if event == '400M Men':
        return [-79, 1.021, 0]
    if event == '800M Men':
        return [-182, 0.198, 0]
    if event == '1500M Men':
        return [-385, 0.04066, 0]
    if event == '5000M Men':
        return [-1440, 0.002778, 0]
    if event == '10000M Men':
        return [-3150, 0.000524, 0]
    if event == 'Marathon Men':
        return [-15600, 0.0000191, 0]
    if event == '110M Hurdles Men':
        return [-25.8, 7.66, 0]
    if event == '400M Hurdles Men':
        return [-95.5, 0.546, 0]
    if event == '3000M Steeplechase Men':
        return [-1020, 0.004316, 0]
    if event == '20Km Race Walk Men':
        return [-11400, 0.00002735, 0]
    if event == '50Km Race Walk Men':
        return [-37200, 0.000002124, 0]
    if event == '4X100M Relay Men':
        return [-69.5, 1.236, 0]
    if event == '4X400M Relay Men':
        return [-334, 0.05026, 0]

    # Men's field events
    if event == 'High Jump Men':
        return [11.534, 32.29, -5000]
    if event == 'Pole Vault Men':
        return [39.39, 3.042, -5000]
    if event == 'Long Jump Men':
        return [48.41, 1.929, -5000]
    if event == 'Triple Jump Men':
        return [98.63, 0.4611, -5000]
    if event == 'Shot Put Men':
        return [687.7, 0.042172, -20000]
    if event == 'Discus Throw Men':
        return [2232.6, 0.004007, -20000]
    if event == 'Hammer Throw Men':
        return [2669.4, 0.0028038, -20000]
    if event == 'Javelin Throw Men':
        return [2886.8, 0.0023974, -20000]

    # Combined
    if event == 'Decathlon Men':
        return [71170, 0.00000097749, -5000]

    # Women's running events
    if event == '100M Women':
        return [-22, 9.92, 0]
    if event == '200M Women':
        return [-45.5, 2.242, 0]
    if event == '400M Women':
        return [-110, 0.335, 0]
    if event == '800M Women':
        return [-250, 0.0688, 0]
    if event == '1500M Women':
        return [-540, 0.0134, 0]
    if event == '5000M Women':
        return [-2100, 0.000808, 0]
    if event == '10000M Women':
        return [-4500, 0.0001712, 0]
    if event == 'Marathon Women':
        return [-22800, 0.00000595, 0]
    if event == '100M Hurdles Women':
        return [-30, 3.98, 0]
    if event == '400M Hurdles Women':
        return [-130, 0.208567, 0]
    if event == '3000M Steeplechase Women':
        return [-1510, 0.001323, 0]
    if event == '20Km Race Walk Women':
        return [-13200, 0.0000187, 0]
    if event == '4X100M Relay Women':
        return [-98, 0.3895, 0]
    if event == '4X400M Relay Women':
        return [-480, 0.01562, 0]

    # Women's field events
    if event == 'High Jump Women':
        return [10.574, 39.34, -5000]
    if event == 'Pole Vault Women':
        return [34.83, 3.953, -5000]
    if event == 'Long Jump Women':
        return [49.24, 1.966, -5000]
    if event == 'Triple Jump Women':
        return [105.53, 0.4282, -5000]
    if event == 'Shot Put Women':
        return [657.53, 0.0462, -20000]
    if event == 'Discus Throw Women':
        return [2227.3, 0.0040277, -20000]
    if event == 'Hammer Throw Women':
        return [2540, 0.0030965, -20000]
    if event == 'Javelin Throw Women':
        return [2214.9, 0.004073, -20000]

    # Combined
    if event == 'Heptathlon Women':
        return [55990, 0.000001581, -5000]

def make_complete_df(df):
    df['Result'] = df['Result'].apply(convert_to_seconds)
    df['Constants'] = df['Event'].apply(get_constants)
    df['IAAF Score'] = df['Result'].apply(get_scores, df=df)

    return df

df_converted = make_complete_df(df)

def performances_over_time():
    """
    Graphs the average IAAF score for each year on a time axis
    Returns the graph as a plotly line graph
    """
    print('Generating Performances Plot...')
    avg_scores = df_converted.groupby('Year')['IAAF Score'].mean()
    avg_scores = pd.DataFrame({'Year': avg_scores.index, 'IAAF Score': avg_scores.values})

    plot = px.line(avg_scores, x='Year', y='IAAF Score')
    plot.update_layout(title=('Overall Performance Over Time'))

    return plot


def gold_medal_map(continent="world"):
    """
    Takes in one of 6 string parameters to choose from:
    "world" | "europe" | "asia" | "africa" | "north america" | "south america"
    and returns a map of the countries from the continent the parameter
    specified, with each country having a shade of color that depends on the
    number of olympic gold medals they earned.

    If no parameter is specfied the default parameter is "world" and will
    return a map of the world with all countries with all of the previously
    mentioned features.
    """
    print('Generating Gold Medal Map...')
    continent_lower = continent.lower()
    if continent_lower == "world":
        cap_continent = 'the ' + continent_lower.capitalize()
    else:
        cap_continent = continent_lower.capitalize()

    only_gold = df[df['Medal'] == 'G']
    total_medal = only_gold.groupby('Nationality')['Medal'].count()
    total_medal_df = total_medal.to_frame()
    total_medals_pre = total_medal_df.reset_index()
    total_gold_medals = total_medals_pre.rename(
                                            columns={'Nationality': 'Country',
                                                     'Medal': 'Gold Medals'})

    fig = px.choropleth(total_gold_medals, locations='Country',
                        color='Gold Medals', scope=continent_lower,
                        color_continuous_scale='spectral')

    fig.update_layout(title=('Gold Medals Per Country in ' + cap_continent))

    return fig


def predict_results(event, medal):
    """
    Takes an event and medal type as a string parameter and uses a decision tree
    regressor to predict the top 3 results given medal and nationality of a
    competitor
    """
    filtered = df_converted[(df_converted['Event'] == event) & (df_converted['Medal'] == medal)]
    filtered = filtered.loc[:, ['Nationality', 'Result']]
    X = filtered.loc[:, filtered.columns != 'Result']
    y = filtered['Result']
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    model = DecisionTreeRegressor()

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    mark = sorted(y_test_pred)[0]

    return mark


def simulate():
    """
    Predicts the top 3 results for every Olympic event
    Returns a dataframe containing the event name, medal type, and mark of
    each result
    """
    print('Predicting Results...')
    event_array = pd.unique(df_converted['Event'])
    medal_list = ['G', 'S', 'B']
    events = []
    medals = []
    marks = []
    temp_marks = []

    for event in event_array:
        temp_marks = []
        for medal in medal_list:
            mark = predict_results(event, medal)

            events.append(event)
            temp_marks.append(mark)

            if (medal == 'G'):
                medals.append('Gold')
            elif (medal == 'S'):
                medals.append('Silver')
            else:
                medals.append('Bronze')


        if event in FIELD:
            ordered = sorted(temp_marks, reverse=True)
        else:
            ordered = sorted(temp_marks)

        marks.extend(ordered)

        sim = pd.DataFrame({'Event': events, 'Medal': medals, 'Result': marks})

    # convert all but multi events to h:mm:ss.ms
    mask = ((sim['Event'] != 'Decathlon Men') & (sim['Event'] != 'Heptathlon Women'))

    sim['Result'] = np.where(mask, sim['Result'].apply(convert_from_seconds), sim['Result'])

    return sim

if __name__ == '__main__':
    main()
