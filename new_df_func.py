def new_dataset(df_orig, colname):
    
    df_new = df_orig.copy()

    def tokenize(x):
        return x.split('///')[1].split('/')

    d_p = df_orig[colname].apply(tokenize)

    df_new['year'] = d_p.apply(lambda x: x[0])
    df_new['season'] = d_p.apply(lambda x: x[1])
    df_new['type'] = d_p.apply(lambda x: x[2])
    df_new['section'] = d_p.apply(lambda x: x[3])

    return df_new