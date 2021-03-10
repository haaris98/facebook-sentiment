import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from sklearn import datasets

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


def main():
    st.title("Sentiment Analysis of Telco in Malaysia using Facebook Comments")
    st.sidebar.title("Sentiment Analysis of Telco in Malaysia using Facebook Comments")
    st.sidebar.title("😀 😐 😡")
    st.sidebar.markdown("This Sentiment Analysis has been done using VADER")
    st.sidebar.markdown("The data that has been preprocessed used in this Data Visualization.")
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("Telco-2020-VADER-Data-Preprocessing.csv")
        data["created_time"] = pd.to_datetime(data["created_time"])
        return data

    data = load_data()

    # Show random tweet
    st.sidebar.subheader("Show Random Facebook Comment")
    random_tweet = st.sidebar.radio("Sentiment", ("positive", "neutral", "negative"))
    if not st.sidebar.checkbox("Hide", True, key='0'):
        st.subheader(f"Random {random_tweet.capitalize()} Facebook Comment")
        st.header(data.query("sentiment == @random_tweet")[["cleanText"]].sample(n=1).iat[0, 0])

    # Number of telco by sentiment
    st.sidebar.subheader("Number of Facebook Comments by Sentiment")
    select = st.sidebar.selectbox("Visualization Type", ["Bar Plot", "Pie Chart"])
    sentiment_count = data["sentiment"].value_counts()
    sentiment_count = pd.DataFrame({"Sentiment":sentiment_count.index, "Comments":sentiment_count.values})
    if not st.sidebar.checkbox("Hide", True, key='1'):
        st.subheader("Number of Facebook Comments by Sentiment")
        if select == "Bar Plot":
            fig = px.bar(sentiment_count, x="Sentiment", y="Comments", color="Comments")
            st.plotly_chart(fig)
        if select == "Pie Chart":
            fig = px.pie(sentiment_count, values="Comments", names="Sentiment")
            st.plotly_chart(fig)

    # Number of tweets for each telco
    st.sidebar.subheader("Number of Facebook Comments for Each Telco")
    each_telco = st.sidebar.selectbox("Visualization Type", ["Bar Plot", "Pie Chart"], key="3")
    telco_sentiment_count = data.groupby("telco")["sentiment"].count().sort_values(ascending=False)
    telco_sentiment_count = pd.DataFrame({"Telco":telco_sentiment_count.index, "Comments":telco_sentiment_count.values.flatten()})
    if not st.sidebar.checkbox("Hide", True, key="4"):
        if each_telco == "Bar Plot":
            st.subheader("Number of Facebook Comments for Each Telco")
            fig = px.bar(telco_sentiment_count, x="Telco", y="Comments", color="Comments", color_discrete_sequence=px.colors.qualitative.D3)
            st.plotly_chart(fig)
        if each_telco == "Pie Chart":
            st.subheader("Number of Facebook Comments for Each Telco")
            fig = px.pie(telco_sentiment_count, values="Comments", names="Telco")
            st.plotly_chart(fig)
    
    # Breakdown telco comments by sentiment
    st.sidebar.subheader("Breakdown Telco Facebook Comments by Sentiment")
    choice = st.sidebar.multiselect("Choose Telco(s)", tuple(pd.unique(data["telco"])))
    if not st.sidebar.checkbox("Hide", True, key="5"):
        st.subheader("Breakdown Telco Facebook Comments by Sentiment")
        if len(choice) > 0:
            chosen_data = data[data["telco"].isin(choice)]
            fig = px.histogram(chosen_data, x="telco", y="sentiment",
                                histfunc="count", color="sentiment",
                                facet_col="sentiment", labels={"sentiment": "sentiment"})
            st.plotly_chart(fig)

    # Word cloud
    st.sidebar.subheader("Word Cloud")
    word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(data["sentiment"])))
    if not st.sidebar.checkbox("Hide", True, key="6"):
        st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
        df = data[data["sentiment"]==word_sentiment]
        words = " ".join(df["lemmatized"])
        processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
        fig, ax = plt.subplots()
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)

    @st.cache
    def load_data(persist=True):
        data = pd.read_csv('Digi-2020-Sentiment.csv') #Telco-2020-Sentiment-TextBlob-No-Neutral.csv
        return data

    df = load_data()

    # Feature Extraction
    # selecting feature extraction 
    st.sidebar.subheader("Feature Extraction")
    feature_name = st.sidebar.selectbox("Select Feature Extraction Methods",("TF-IDF","Bag-Of-Words"))
    if not st.sidebar.checkbox("Hide", True, key="7"):
        st.title(feature_name)
        st.subheader("Running..")
        # TF-IDF
        if(feature_name == "TF-IDF"):
            vectorizer = TfidfVectorizer (max_features=100, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
            processed_features = vectorizer.fit_transform(df['sentence']).toarray()
            st.subheader("Done!")
            # Display Horizontal Bar Chart
            neg_matrix = vectorizer.transform(df[df.sentiment == 0].sentence)
            pos_matrix = vectorizer.transform(df[df.sentiment == 1].sentence)
            neg_words = neg_matrix.sum(axis=0)
            neg_words_freq = [(word, neg_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            neg_tf = pd.DataFrame(list(sorted(neg_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','negative'])
            neg_tf_df = neg_tf.set_index('Terms')

            pos_words = pos_matrix.sum(axis=0)
            pos_words_freq = [(word, pos_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            pos_words_tf = pd.DataFrame(list(sorted(pos_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','positive'])
            pos_words_tf_df = pos_words_tf.set_index('Terms')

            # Top positive 20 Terms by TF-IDF weighting
            fig1 = plt.figure()
            y_pos = np.arange(20)
            plt.barh(y_pos, pos_words_tf_df.sort_values(by='positive', ascending=False)['positive'][:20], align='center', alpha=0.5)
            plt.yticks(y_pos, pos_words_tf_df.sort_values(by='positive', ascending=False)['positive'][:20].index)
            plt.ylabel('Positive Terms')
            plt.xlabel('Weight')
            plt.title('Top 20 Positive Terms using TF-IDF')
            st.pyplot(fig1)

            # Top negative 20 Terms by TF-IDF weighting
            fig2 = plt.figure()
            y_pos = np.arange(20)
            plt.barh(y_pos, neg_tf_df.sort_values(by='negative', ascending=False)['negative'][:20], align='center', color="red", alpha=0.5)
            plt.yticks(y_pos, neg_tf_df.sort_values(by='negative', ascending=False)['negative'][:20].index)
            plt.ylabel('Negative Terms')
            plt.xlabel('Weight')
            plt.title('Top 20 Negative Terms using TF-IDF')
            st.pyplot(fig2)
        # Bag-of-words
        else:
            cv = CountVectorizer(stop_words='english',max_features=100)
            processed_features = cv.fit_transform(df['sentence']).toarray()
            st.subheader("Done!")
            # Display Horizontal Bar Chart
            neg_matrix = cv.transform(df[df.sentiment == 0].sentence)
            pos_matrix = cv.transform(df[df.sentiment == 1].sentence)

            neg_words = neg_matrix.sum(axis=0)
            neg_words_freq = [(word, neg_words[0, idx]) for word, idx in cv.vocabulary_.items()]
            neg_tf = pd.DataFrame(list(sorted(neg_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','negative'])
            neg_tf_df = neg_tf.set_index('Terms')

            pos_words = pos_matrix.sum(axis=0)
            pos_words_freq = [(word, pos_words[0, idx]) for word, idx in cv.vocabulary_.items()]
            pos_words_tf = pd.DataFrame(list(sorted(pos_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','positive'])
            pos_words_tf_df = pos_words_tf.set_index('Terms')

            # Top positive terms BOW
            fig1 = plt.figure()
            y_pos = np.arange(20)
            plt.barh(y_pos, pos_words_tf_df.sort_values(by='positive', ascending=False)['positive'][:20], align='center', alpha=0.5)
            plt.yticks(y_pos, pos_words_tf_df.sort_values(by='positive', ascending=False)['positive'][:20].index,rotation=45)
            plt.ylabel('Top 20 Positive Terms')
            plt.xlabel('Frequency')
            plt.title('Top 20 Positive Terms using Bag-of-Word')
            st.pyplot(fig1)
            # Top negative terms BOW
            fig2 = plt.figure()
            y_pos = np.arange(20)
            plt.barh(y_pos, neg_tf_df.sort_values(by='negative', ascending=False)['negative'][:20], align='center', color="red", alpha=0.5)
            plt.yticks(y_pos, neg_tf_df.sort_values(by='negative', ascending=False)['negative'][:20].index,rotation=45)
            plt.ylabel('Top 20 Negative Terms')
            plt.xlabel('Frequency')
            plt.title('Top 20 Negative Terms using Bag-of-Word')
            st.pyplot(fig2)
    

    # selecting classifier 
    st.sidebar.subheader("Machine Learning Algorithms")
    classifier_name = st.sidebar.selectbox("Select Classifier",("Random Forest", "Decision Tree","Logistic Regression","Naive Bayes","Support Vector Machine"))
    if not st.sidebar.checkbox("Hide", True, key="8"):
    
        # Divide data into training and test sets

        labels = np.array(df['sentiment'])

        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.3, random_state=69)
        # Classifier
        if(classifier_name == "Random Forest"):
            st.title(classifier_name)
            st.subheader("Running..")
            rf = RandomForestClassifier(n_estimators=200, random_state=5)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            st.subheader("Done!")
            st.write("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
            st.write("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
            
            acc = rf.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            cm=confusion_matrix(y_test, y_pred)
            st.write('Confusion Matrix: ', cm)
            st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
            st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
            st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
            st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
            # Calculate AUC
            prob_RF = rf.predict_proba(X_test)
            prob_RF = prob_RF[:, 1]

            auc_RF = roc_auc_score(y_test, prob_RF)
            st.write('AUC : %.2f' % auc_RF)

            # Plot ROC Curve
            fig = plt.figure()
            fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF) 
            plt.plot(fpr_RF, tpr_RF, color='blue', label='RF') 
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            st.pyplot(fig)

            # Plot Precision-Recall Curve
            fig1 = plt.figure()
            prec_RF, rec_RF, threshold_RF = precision_recall_curve(y_test, prob_RF)
            plt.plot(prec_RF, rec_RF, color='blue', label='RF') 
            plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            st.pyplot(fig1)

        elif(classifier_name == "Decision Tree"):
            st.title(classifier_name)
            st.subheader("Running..")
            DT = DecisionTreeClassifier(max_depth=50)
            DT.fit(X_train, y_train)
            y_pred = DT.predict(X_test)
            st.subheader("Done!")
            st.write("Accuracy on training set: {:.3f}".format(DT.score(X_train, y_train)))
            st.write("Accuracy on test set: {:.3f}".format(DT.score(X_test, y_test)))
            
            acc = DT.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            cm=confusion_matrix(y_test, y_pred)
            st.write('Confusion Matrix: ', cm)
            st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
            st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
            st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))

            # Calculate AUC
            prob_DT = DT.predict_proba(X_test)
            prob_DT = prob_DT[:, 1]

            auc_DT = roc_auc_score(y_test, prob_DT)
            st.write('AUC : %.2f' % auc_DT)

            # Plot ROC Curve
            fig = plt.figure()
            fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT) 
            plt.plot(fpr_DT, tpr_DT, color='aqua', label='DT') 
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()

            st.pyplot(fig)

            # Plot Precision-Recall Curve
            fig1 = plt.figure()
            prec_DT, rec_DT, threshold_DT = precision_recall_curve(y_test, prob_DT)
            plt.plot(prec_DT, rec_DT, color='aqua', label='DT') 
            plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

            st.pyplot(fig1)

        elif(classifier_name == "Logistic Regression"):
            st.title(classifier_name)
            st.subheader("Running..")
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            st.subheader("Done!")
            st.write("Accuracy on training set: {:.3f}".format(lr.score(X_train, y_train)))
            st.write("Accuracy on test set: {:.3f}".format(lr.score(X_test, y_test)))

            acc = lr.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            cm=confusion_matrix(y_test, y_pred)
            st.write('Confusion Matrix: ', cm)
            st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
            st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
            st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))

            # Classification report
            #st.write(classification_report(y_test,y_pred))

            # Calculate AUC
            prob_LR = lr.predict_proba(X_test)
            prob_LR = prob_LR[:, 1]

            auc_LR= roc_auc_score(y_test, prob_LR)
            st.write('AUC : %.2f' % auc_LR)

            # Plot ROC Curve
            fig = plt.figure()
            fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, prob_LR) 
            plt.plot(fpr_LR, tpr_LR, color='red', label='LR') 
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()

            st.pyplot(fig)

            # Plot Precision-Recall Curve
            fig1 = plt.figure()
            prec_LR, rec_LR, threshold_LR = precision_recall_curve(y_test, prob_LR)
            plt.plot(prec_LR, rec_LR, color='red', label='LR') 
            plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

            st.pyplot(fig1)

        elif(classifier_name == "Naive Bayes"):
            st.title(classifier_name)
            st.subheader("Running..")
            nb = MultinomialNB() 
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            st.subheader("Done!")
            st.write("Accuracy on training set: {:.3f}".format(nb.score(X_train, y_train)))
            st.write("Accuracy on test set: {:.3f}".format(nb.score(X_test, y_test)))

            acc = nb.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            cm=confusion_matrix(y_test, y_pred)
            st.write('Confusion Matrix: ', cm)
            st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
            st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
            st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))

            # Classification report
            #st.write(classification_report(y_true=y_test,y_pred=y_pred))

            # Calculate AUC
            prob_NB = nb.predict_proba(X_test)
            prob_NB = prob_NB[:, 1]

            auc_NB= roc_auc_score(y_test, prob_NB)
            st.write('AUC : %.2f' % auc_NB)

            # Plot ROC Curve
            fig = plt.figure()
            fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB) 
            plt.plot(fpr_NB, tpr_NB, color='orange', label='NB') 
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()

            st.pyplot(fig)

            # Plot Precision-Recall Curve
            fig1 = plt.figure()
            prec_NB, rec_NB, threshold_NB = precision_recall_curve(y_test, prob_NB)
            plt.plot(prec_NB, rec_NB, color='orange', label='NB') 
            plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

            st.pyplot(fig1)
        else:
            st.title(classifier_name)
            st.subheader("Running..")
            svc = SVC(kernel='linear', gamma='auto') # 'linear'kernel='rbf'
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            st.subheader("Done!")
            st.write("Accuracy on training set: {:.3f}".format(svc.score(X_train, y_train)))
            st.write("Accuracy on test set: {:.3f}".format(svc.score(X_test, y_test)))

            acc = svc.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            cm=confusion_matrix(y_test, y_pred)
            st.write('Confusion Matrix: ', cm)
            st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
            st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
            st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
            st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

            # Classification report
            #st.write(classification_report(y_test,y_pred))

if __name__ == "__main__":
    main()