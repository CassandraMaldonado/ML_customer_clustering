import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st

df = pd.read_csv("Ecommerce data_customer_cluster.csv")

_df = df[['cluster_number','sub_cluster','Ffiction1', 'Fclassics3',
       'Fcartoons5', 'Flegends6', 'Fphilosophy7', 'Freligion8', 'Fpsychology9',
       'Flinguistics10', 'Fart12', 'Fmusic14', 'Ffacsimile17', 'Fhistory19',
       'Fconthist20', 'Feconomy21', 'Fpolitics22', 'Fscience23', 'Fcompsci26',
       'Frailroads27', 'Fmaps30', 'Ftravelguides31', 'Fhealth35', 'Fcooking36',
       'Flearning37', 'FGamesRiddles38', 'Fsports39', 'Fhobby40', 'Fnature41',
       'Fencyclopaedia44', 'Fvideos50', 'Fnonbooks99']].groupby(['cluster_number','sub_cluster']).sum().reset_index()

cat = ['Ffiction1', 'Fclassics3',
       'Fcartoons5', 'Flegends6', 'Fphilosophy7', 'Freligion8', 'Fpsychology9',
       'Flinguistics10', 'Fart12', 'Fmusic14', 'Ffacsimile17', 'Fhistory19',
       'Fconthist20', 'Feconomy21', 'Fpolitics22', 'Fscience23', 'Fcompsci26',
       'Frailroads27', 'Fmaps30', 'Ftravelguides31', 'Fhealth35', 'Fcooking36',
       'Flearning37', 'FGamesRiddles38', 'Fsports39', 'Fhobby40', 'Fnature41',
       'Fencyclopaedia44', 'Fvideos50', 'Fnonbooks99']


_df['Ftotal'] = _df.sum(axis=1)

for c in cat:
    _df[c]=_df[c]/_df['Ftotal']

rec_table = _df

st.sidebar.header('Select User')
option = st.sidebar.selectbox(
'What user are you making the recommendation?',
 df['id'])


# df.loc[df['id']==914,'sub_cluster'].astype(int).values

selected_cluster_number = df.loc[df['id']==option,'sub_cluster'].astype(int).values[0]
selected_sub_cluster = df.loc[df['id']==option,'cluster_number'].astype(int).values[0]

st.sidebar.write('Cluster: ', selected_cluster_number)
st.sidebar.write('Sub Cluster: ', selected_sub_cluster)

rec_table.drop('Ftotal',axis=1, inplace=True)
result = rec_table[(rec_table['cluster_number'].astype(int)==selected_cluster_number) & (rec_table['sub_cluster'].astype('int')==selected_sub_cluster)].melt().sort_values(by='value', ascending=False)[2:5]

result

# 3 Columns

item_dict = {'Ffiction1':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiblart.com%2Fwp-content%2Fuploads%2F2020%2F01%2Fcrime-and-mystery-cover-scaled-1.jpeg&f=1&nofb=1&ipt=4e250243340f68841bbeacb410519d5cc49127cea44e3630704960a5a60e74f5&ipo=images'
             , 'Fclassics3':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fs-i.huffpost.com%2Fgen%2F1148926%2Fimages%2Fo-CLASSIC-BOOKS-ORIGINAL-COVERS-facebook.jpg&f=1&nofb=1&ipt=01183ebe92ce151d941ad45ae3c9cc3aeb7c2a29c775c54bd16af62f1bcbe66e&ipo=images'
             , 'Fcartoons5':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.vectorstock.com%2Fi%2F1000x1000%2F76%2F01%2Fcartoon-smiling-book-vector-2087601.jpg&f=1&nofb=1&ipt=8cca6c82ce4c51b49a6b1a65eb6d19ee5c2c010b9a21152f52d2f87656784184&ipo=images'
             , 'Flegends6':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fd28hgpri8am2if.cloudfront.net%2Fbook_images%2Fonix%2Fcvr9780857758491%2Fmyths-legends-9780857758491_hr.jpg&f=1&nofb=1&ipt=474dc087b572ec96b94c15c6a6ae8ca9e6cb9616782be3d866eb614dac030bb1&ipo=images'
             , 'Fphilosophy7':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fd28hgpri8am2if.cloudfront.net%2Fbook_images%2Fonix%2Fcvr9781440567674%2Fphilosophy-101-9781440567674_hr.jpg&f=1&nofb=1&ipt=318c549e8bcf378e76ae18989ff880d9c7036dcb387058b87593302571584e92&ipo=images'
             , 'Freligion8':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fd28hgpri8am2if.cloudfront.net%2Fbook_images%2Fonix%2Fcvr9781440572630%2Freligion-101-9781440572630_hr.jpg&f=1&nofb=1&ipt=6e94c6ba9d571b48ad13eef150ab89445070a53a4fd37162736e8bd012e7bcf6&ipo=images'
             , 'Fpsychology9':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.sherwoodbooks.co.za%2Fwp-content%2Fuploads%2F2020%2F07%2F9780190404697-6.jpg&f=1&nofb=1&ipt=c68b53456fa93f9a70a85bea96363fa42cb5fd0b1e73690040c53c5ffdc4c49b&ipo=images'
             , 'Flinguistics10':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.penguin.com.au%2Fcovers%2Foriginal%2F9780262533263.jpg&f=1&nofb=1&ipt=d6d59fee7b82011523ac2419251cdba1b7dfb9a0049c3bd42e6edfe36d32e88f&ipo=images'
             , 'Fart12':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fs-i.huffpost.com%2Fgen%2F906538%2Fimages%2Fo-BEST-ART-BOOKS-2012-facebook.jpg&f=1&nofb=1&ipt=39cfbf75d4b67910fb657d0528babfb71b425dfc2a5b43db2603cf55e8dcd779&ipo=images'
             , 'Fmusic14':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fthebookcoverdesigner.com%2Fwp-content%2Fuploads%2F2015%2F09%2FMusic-scaled.jpg&f=1&nofb=1&ipt=d6da0abb98b43c434dbe6f5ab1304b6bb906e1916d5a4caaafe67f11bfb3a77f&ipo=images'
             , 'Ffacsimile17':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fhistoryofinformation.com%2Fimages%2FDomesday_Book_cloth_upper_cover_big.jpeg&f=1&nofb=1&ipt=3fa4c1b4492be9eee5a6301675456e0ee93d384233993ea9e35d6f2635b3b1c2&ipo=images'
             , 'Fhistory19':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.penguin.com.au%2Fcovers%2Foriginal%2F9780241225929.jpg&f=1&nofb=1&ipt=432b5ddd999bd93acfe24572607845e632a3d1681d5adeac50643651318940dc&ipo=images'
             , 'Fconthist20':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.vectorstock.com%2Fi%2F1000x1000%2F76%2F01%2Fcartoon-smiling-book-vector-2087601.jpg&f=1&nofb=1&ipt=8cca6c82ce4c51b49a6b1a65eb6d19ee5c2c010b9a21152f52d2f87656784184&ipo=images'
             , 'Feconomy21':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.bibdsl.co.uk%2Fimagegallery2%2Fbds%2F201236%2F9781409376415.JPG&f=1&nofb=1&ipt=90dcef66df41e5dd942279323fb7b8f3b4a00fa90ef6756aa37984a90b338565&ipo=images'
             , 'Fpolitics22':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.penguin.com.au%2Fcovers%2Foriginal%2F9781409364450.jpg&f=1&nofb=1&ipt=fdf91cfe53c95305e1633fc19ccebd687a7977b8eb8bd4470deb96e2b5226d06&ipo=images'
             , 'Fscience23':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn2.penguin.com.au%2Fcovers%2Foriginal%2F9780241317815.jpg&f=1&nofb=1&ipt=4ea90e821ca5e990e843ba3c65b297993b6f80813185798dca6b40a3410d951a&ipo=images'
             , 'Fcompsci26':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.hackr.io%2Fuploads%2Fposts%2Fattachments%2F16431216080SuUqgLuUG.png&f=1&nofb=1&ipt=9fcc9b4d0c6ef36ba0848ae8e285f69efa5efe99b4106ffbe055c27bbc834e21&ipo=images'
             , 'Frailroads27':'https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fstore-gsrj-com.3dcartstores.com%2Fassets%2Fimages%2FRailroads%2520Across%2520North%2520America.jpg&f=1&nofb=1&ipt=3ec041cfb0ebc307dc51cbc7b2915088328c036ea435f7a96b291a85f8b14a94&ipo=images'
             , 'Fmaps30':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F2f96be1b505f7f7a63c3-837c961929b51c21ec10b9658b068d6c.ssl.cf2.rackcdn.com%2Fproducts%2F043296.jpg&f=1&nofb=1&ipt=3d0a0faa52a7cd74be1405ab9465425079f830afa56604cb003552c5d79f35c9&ipo=images'
             , 'Ftravelguides31':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fhachette.imgix.net%2Fbooks%2F9781640497702.jpg%3Fauto%3Dcompress%2Cformat&f=1&nofb=1&ipt=553d7068f8ddd3bec9af4a7105c6629ab189c99b70dadd717b0bde9b37dd3c8f&ipo=images'
             , 'Fhealth35':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.g-w.com%2Fassets%2Fimages%2Fbookmd%2F9781635630305.jpg&f=1&nofb=1&ipt=4a1202d002d4910fa543644297a3002bc6a7c896aacd035d8bc0e32eb29c2108&ipo=images'
             , 'Fcooking36':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.shopify.com%2Fs%2Ffiles%2F1%2F0077%2F9010%2F0553%2Fproducts%2Fmarissasbooksandgifts-9781760790790-the-basic-cookbook-guide-30529042481351_800x.jpg%3Fv%3D1628377460&f=1&nofb=1&ipt=79e3576e509e6c053595b9008597cdd260a6cbcd8bb0f19b9402ddb922e138d5&ipo=images'
             , 'Flearning37':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.schandpublishing.com%2Fuploads%2Fbookimages%2Fschand-books%2F9789325978461.jpg&f=1&nofb=1&ipt=df177fc51cebe1f69b82024956c63b4bbc5434eba75b1ba60452f2ade06255b4&ipo=images'
             , 'FGamesRiddles38':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi5.walmartimages.com%2Fasr%2Fb2a61770-4ac6-4d05-891a-41d91d3c49ec_1.69a9bee168a3d2e42752eadf169d2648.jpeg%3FodnWidth%3D612%26odnHeight%3D612%26odnBg%3Dffffff&f=1&nofb=1&ipt=ae9c7cd97ecbba52c5e6009b0bd306b5259da562e24cf882093f93673dd0c95d&ipo=images'
             , 'Fsports39':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi5.walmartimages.com%2Fasr%2F566070a7-533f-459a-8a95-90e6b2309d07.b5045e1ddafa331217066dc599dad104.jpeg%3FodnWidth%3D1000%26odnHeight%3D1000%26odnBg%3Dffffff&f=1&nofb=1&ipt=11ff9cbccba49f767b8b84b4f265ddb4279baf8f3bcbc65a329cfd06cc658491&ipo=images'
             , 'Fhobby40':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.pinimg.com%2Foriginals%2Fe8%2F38%2Fe8%2Fe838e8db5569087bd203b8ac94ff0a22.jpg&f=1&nofb=1&ipt=a93fa759946d9dd41daa33915002a82f09493e0ad2a983d24bae8c3866666251&ipo=images'
             , 'Fnature41':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi5.walmartimages.com%2Fasr%2Fc7d28019-3921-42e0-b65a-074d910c2f22_4.3bac17c96e8d54d09b9a8ee44db09634.jpeg&f=1&nofb=1&ipt=4b7a2fca35cabe07beb0c25e51f8b4c560f286d1b5fcc7535e19e01d358aa7a7&ipo=images'
             , 'Fencyclopaedia44':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fn1.sdlcdn.com%2Fimgs%2Fb%2Fl%2Fx%2FEncyclopaedia-Britannica-SDL642339200-1-434a8.jpg&f=1&nofb=1&ipt=4c5546202135b0b043b367db3ddf25a3e9da013fe76cfde8d4f817b26eaba62c&ipo=images'
             , 'Fvideos50':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F000%2F646%2F238%2Foriginal%2Fvector-video-icon-symbol-sign.jpg&f=1&nofb=1&ipt=6417cde0fc8611e3366f4b79f7a1b9c02f3e509e94fe7071b9c0a803166a8e75&ipo=images'
             , 'Fnonbooks99':'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn5.vectorstock.com%2Fi%2F1000x1000%2F70%2F99%2Fno-book-sign-icon-open-book-symbol-vector-1907099.jpg&f=1&nofb=1&ipt=5246508615d10f08edbb1d04aa24868d71cffba40fe9feeb1b8c1bab4584b2d2&ipo=images'}

col1, col2, col3 = st.columns(3)
with col1:
  st.header(result.iloc[0,0])
  st.image(item_dict[result.iloc[0,0]], use_column_width=True)
with col2:
  st.header(result.iloc[1,0])
  st.image(item_dict[result.iloc[1,0]], use_column_width=True)
with col3:
  st.header(result.iloc[2,0])
  st.image(item_dict[result.iloc[2,0]], use_column_width=True)