from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
import numpy as np

def ordinal_encoding():
    data = [['US'],['UK'],['NZ'],['IR']]
    print(data)
    
    # define encoder
    encoder = OrdinalEncoder()
    
    # our label feature
    encoded_data = encoder.fit_transform(data)
    print(encoded_data)


def label_encoding():
    data = ['US','UK','NZ','IR']
    print(data)
    
    # define encoder
    encoder = LabelEncoder()
    
    # our label feature
    encoded_data = encoder.fit_transform(data)
    print(encoded_data)
    
    
def one_hot_encoding():
    enc = OneHotEncoder(handle_unknown='ignore')
    data = np.asarray([['US'], ['UK'], ['NZ']])
    enc.fit(data)
    print(enc.categories_)
    onehotlabels = enc.transform(data).toarray() 
    print(onehotlabels)
       
ordinal_encoding()
label_encoding()
one_hot_encoding()