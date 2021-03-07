def clean(data):
    """
    removes all rows with empty data in the Results column and all useless
    columns
    fixes some of the time formatting for weird times
    """

    data['Result'] = data['Result'].astype(str)
    data = data[data['Result'] != 'None']
    data = data.drop(['Unnamed: 8'], axis=1)
    data['Result'] = data['Result'].str.replace('h', ':')
    data['Result'] = data['Result'].str.replace('est', '')
    data['Result'] = data['Result'].str.replace('-', ':')
    data['Result'] = data['Result'].str.replace('P.', '')
    data['Result'] = data['Result'].str.strip()

    return data
