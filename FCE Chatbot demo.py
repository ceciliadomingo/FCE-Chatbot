import time
from textblob import TextBlob
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

#* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►
#ENTITIES

#To store the sentiment towards each topic
OPINION_TOPIC = {'shops':'NEU',
                 'flats':'NEU',
                 'parks':'NEU',
                 'cameras':'NEU',
                 'clubs':'NEU'}

#To record when reasons have been given to support/reject a topic
REASONED_TOPIC = {'shops':'not_reasoned',
                 'flats':'not_reasoned',
                 'parks':'not_reasoned',
                 'cameras':'not_reasoned',
                 'clubs':'not_reasoned'}

#To record which topics have been discussed
COVERED_TOPIC = {'shops':'not_covered',
                 'flats':'not_covered',
                 'parks':'not_covered',
                 'cameras':'not_covered',
                 'clubs':'not_covered'}
topics_covered = set()
topics_NOTcovered = set()

#To keep a log of what the user says and later analyze it to give feedback
user_utterances = set()



#* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►

#FUNCTIONS

#-----------------------------------------------------------

#Function to process the user's sentence for classification

def Preprocess(text):
    #Setting up NLP tools
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    lemmer = nltk.stem.WordNetLemmatizer()

    #Tokenizing
    word_tokens = tokenizer.tokenize(text)

    #Removing stop words and lemmatizing
    nostopw_sentence = []
    for w in word_tokens: 
        if w.lower() not in stop_words: 
            nostopw_sentence.append(w.lower())
    processed_sentence = []
    for w in nostopw_sentence:
        lemmer.lemmatize(w)
        processed_sentence.append(w)
    return(str(processed_sentence))

#-----------------------------------------------------------

#Function to detect the topic of a sentence

def Classify(newtext): #depends on preprocess
    #Keywords representing each category
    patternSHOP = re.findall(r'(shop|store|buy|boutique|retail).*', Preprocess(newtext), re.IGNORECASE)
    patternFLATS = re.findall(r'(apartment|flat|accommodat|lodg|hotel|house|hostel).*', Preprocess(newtext), re.IGNORECASE)
    patternPARK = re.findall(r'(park|natur|tree|flower|picnic|pic\-nic|green|animal|wildlife).*', Preprocess(newtext), re.IGNORECASE)
    patternCAMERA = re.findall(r'(camera|record|film|securit|safet|spy|spies|crime|criminal|pick pocket|pickpocket|thief|thieves|steal|robb).*', Preprocess(newtext), re.IGNORECASE)
    patternCLUB = re.findall(r'(club|party|partie|disco|night life|music|danc).*', Preprocess(newtext), re.IGNORECASE)

    #The score for each category is the number of matches of the pattern
    classification_score = {'shops':len(patternSHOP),'flats':len(patternFLATS),'parks':len(patternPARK),'cameras':len(patternCAMERA),'clubs':len(patternCLUB)}

    #The chosen category is the one with the most matches
    return(max(classification_score, key=classification_score.get))

 
#-----------------------------------------------------------

#Patterns to detect connectors 

patternADDING = r"moreover|further.*|what is more|what's more|in addition|besides|not to mention|in fact|actually|in reality|as well as|not only|another|also|too|(?:with (?:respect |regard )to)|concerning|regarding|as for|(?:(?:talking |speaking )about)"
patternEQUATING = r"likewise|similarly|equally|the same way|as well as|(?:(?:coupled |together )with)"
patternORDERING = r"(?:(?:first|second|third|last)(?:ly)*)|next|to begin"
patternCONTRASTING = r"but|however|(?:(?:al)*though)|(?:(?:none|never)theless)|notwithstanding|despite|in spite of|having said that|that being said|(?:on the (?:one |other )hand)|alternatively|whereas|while|at the same time|contrast.*|compar.*|yet"
patternCONSIDERING = r"with this in mind|provided that|(?:in (?:view |light )of)|considering|all things considered|ultimately|at the end of the day|above all"
patternEXAMPLES = r"for instance|example|illustrate|illustration|namely|especially|particularly|in particular|in fact|such as|(?<!people )(?<!they )(?<!I )(?<!don't )(?<!not )like"
patternPARAPHRASING = r"in other words|(?:(?:to put|putting) it another way)|simply|that is to say|meaning that"
patternCONCLUSION = r"conclu.*|to sum up|summariz.*|finally|lastly|in short|on the whole|in brief"
patternLOGIC = r"therefore|thus|hence|(?: as a (?:result|consequence))|due to|because|(?:for (?:the |this |that )reason)|it follows that|this suggests|logically|of course|naturally|obviously|foregone conclusion|(?:, (?:so|then))"
patternPERSPECTIVES = r"(?:have|make) a .* point|according to|quoting|to quote|(?:the |their |s )(?:point of view|perspective|belief)|are right|differently|(?:(?:(?:(?:some |those )people )|opponents |they |people )(?:who )*(?:may |might |would |could |will )*(?:oppose|disagree|agree|say|think|maintain|claim|believe|consider|(?:(?:dis|don't |not )*like)|are against))"

#-----------------------------------------------------------

#Function to see whether an utterance has connectors

def ClassifyREASON(utterance):
    FINDpatternADDING = re.findall(patternADDING, utterance, re.IGNORECASE)
    FINDpatternEQUATING = re.findall(patternEQUATING, utterance, re.IGNORECASE)
    FINDpatternORDERING = re.findall(patternORDERING, utterance, re.IGNORECASE)
    FINDpatternCONTRASTING = re.findall(patternCONTRASTING, utterance, re.IGNORECASE)
    FINDpatternCONSIDERING = re.findall(patternCONSIDERING, utterance, re.IGNORECASE)
    FINDpatternEXAMPLES = re.findall(patternEXAMPLES, utterance, re.IGNORECASE)
    FINDpatternPARAPHRASING = re.findall(patternPARAPHRASING, utterance, re.IGNORECASE)
    FINDpatternCONCLUSION = re.findall(patternCONCLUSION, utterance, re.IGNORECASE)
    FINDpatternLOGIC = re.findall(patternLOGIC, utterance, re.IGNORECASE)
    FINDpatternPERSPECTIVES = re.findall(patternPERSPECTIVES, utterance, re.IGNORECASE)

    #The score is the total number of matches of the patterns
    TOTALconnector_score = len(FINDpatternADDING) + len(FINDpatternEQUATING) + len(FINDpatternORDERING) + len(FINDpatternCONTRASTING)\
                           + len(FINDpatternCONSIDERING) + len(FINDpatternEXAMPLES) + len(FINDpatternPARAPHRASING) \
                           + len(FINDpatternCONCLUSION) + len(FINDpatternLOGIC) + len(FINDpatternPERSPECTIVES)
    return(TOTALconnector_score)


#-----------------------------------------------------------

#Function to see which connectors the student repeated

def Repetitive_connector(LIST):
    all_connectors_list = []
    used_repetitively = []

    #A score is computed for each utterance of the total and added to the general record
    for utterance in LIST:
        FINDpatternADDING = re.findall(patternADDING, utterance, re.IGNORECASE)
        FINDpatternEQUATING = re.findall(patternEQUATING, utterance, re.IGNORECASE)
        FINDpatternORDERING = re.findall(patternORDERING, utterance, re.IGNORECASE)
        FINDpatternCONTRASTING = re.findall(patternCONTRASTING, utterance, re.IGNORECASE)
        FINDpatternCONSIDERING = re.findall(patternCONSIDERING, utterance, re.IGNORECASE)
        FINDpatternEXAMPLES = re.findall(patternEXAMPLES, utterance, re.IGNORECASE)
        FINDpatternPARAPHRASING = re.findall(patternPARAPHRASING, utterance, re.IGNORECASE)
        FINDpatternCONCLUSION = re.findall(patternCONCLUSION, utterance, re.IGNORECASE)
        FINDpatternLOGIC = re.findall(patternLOGIC, utterance, re.IGNORECASE)
        FINDpatternPERSPECTIVES = re.findall(patternPERSPECTIVES, utterance, re.IGNORECASE)
        all_connectors_list += FINDpatternADDING + FINDpatternEQUATING + FINDpatternORDERING + FINDpatternCONTRASTING \
                           + FINDpatternCONSIDERING + FINDpatternEXAMPLES + FINDpatternPARAPHRASING \
                           + FINDpatternCONCLUSION + FINDpatternLOGIC
    #I count how many times each matched connector is repeated
    repetitions_COUNTER = Counter(all_connectors_list).items()

    #I look at the number part of the counter tuple and return a warninig when the value is high (=repetitiveness)
    for repeatedword in repetitions_COUNTER:
        if repeatedword[1] > 3:
            used_repetitively.append(repeatedword)
    if len(used_repetitively) > 0:
        print("You've been repeating some connectors:\n" + str(used_repetitively) + "\n" \
              + "Using connectors is good, but try to add some variety!\n")

#-----------------------------------------------------------

#Function to see if a statement contains an opinion

def chat_hi_hasopinion(utterance): #depends on Classify
    hasopinion = 0

    #The chatbot keeps asking until an opinion is given
    while hasopinion == 0:
        #Convert utterance to TextBlob
        blobbeduterance = TextBlob(utterance)

        #Classify topic to know in which key of the opinion dictionary to store the opinion
        string_topic = str(Classify(utterance))

        #Sentiment analysis
        sentimiento = float(blobbeduterance.sentiment.polarity)
        if sentimiento > 0:
            OPINION_TOPIC[string_topic] = 'POS'
            hasopinion = 1 #loop ends
        if sentimiento == 0:
            OPINION_TOPIC[string_topic] = 'NEU'
            #Bot asks again
            utterance = input("I see, but tell me in detail what you think about that idea.\n")
        if sentimiento < 0:
            OPINION_TOPIC[string_topic] = 'NEG'
            hasopinion = 1 #loop ends
    user_utterances.add(utterance)

#-----------------------------------------------------------

#Function to see if an opinion is backed by reasons
    
def chat_hasreason(utterance): #depends on ClassifyREASON & Preprocess & Classify
    
    
    #The chatbot keeps asking until a reason is given for the opinion
    reasoned = 0
    while reasoned == 0:
        #Classify topic to know in which key of the reasoning dictionary to store the completion value
        string_topic = str(Classify(utterance))
        reasoning_score = ClassifyREASON(utterance)
        if reasoning_score > 0:
            REASONED_TOPIC[string_topic] = 'reasoned'
            #We update the dictionary so that the Topicscovered function knows not to suggest this topic
            COVERED_TOPIC[string_topic] = 'covered' 
            print("I think you make a good point")
            reasoned = 1 #loop ends
        if reasoned == 0 :
            #The bot asks again
            utterance = input("Uhum, why do you think that?\n")
    user_utterances.add(utterance)

#-----------------------------------------------------------

#Function to suggest a new topic

def New_topic_ask_opinion():
    
    suggestion = random.sample(topics_NOTcovered, 1)
    print("We haven't looked at the topic of " + str(suggestion[0]))


#-----------------------------------------------------------

#Function to detect contradictions or omissions in the conclusion

def EvaluateConclusion(conclusion): #depends on Preprocess
    #Keywords representing each category, to classify the utterance
    patternSHOP = re.findall(r'(shop|store|buy|boutique|retail).*', Preprocess(conclusion), re.IGNORECASE)
    patternFLATS = re.findall(r'(apartment|flat|accommodat|lodg|hotel|house|hostel).*', Preprocess(conclusion), re.IGNORECASE)
    patternPARK = re.findall(r'(park|natur|tree|flower|picnic|pic\-nic|green|animal|wildlife).*', Preprocess(conclusion), re.IGNORECASE)
    patternCAMERA = re.findall(r'(camera|record|film|securit|safet|spy|spies|crime|criminal|pick pocket|pickpocket|thief|thieves|steal|robb).*', Preprocess(conclusion), re.IGNORECASE)
    patternCLUB = re.findall(r'(club|party|partie|disco|night life|music|danc).*', Preprocess(conclusion), re.IGNORECASE)

    #First I detect the topics covered in the conclusion (not with the Classify function, which returns only one)
    classification_score = {'shops':len(patternSHOP),'flats':len(patternFLATS),'parks':len(patternPARK),'cameras':len(patternCAMERA),'clubs':len(patternCLUB)}
    conclusion_topics_list = [] #I store the possibly multiple topics in this list
    for key, value in classification_score.items(): 
         if value > 0: 
             conclusion_topics_list.append(key)
 
    #I compare the list with topics and opinions stored in an entity dictionary to define the incoherence and incompleteness
    incoherence = 0
    incompleteness = 0
    for key, value in OPINION_TOPIC.items():
        #Checking coherence
        if key in conclusion_topics_list:
            if value == 'NEG':
                incoherence = 1
        #Checking completeness
        if (value == 'POS') and (key not in conclusion_topics_list):
            incompleteness = 1

    #Defining the feedback
    if incoherence == 0 and incompleteness == 0:
        print("That's right! Good discussion :) I'll give you some feedback now")
        user_utterances.add(conclusion)
        conclusion_valid.append("Conclusion is valid")
    if incoherence == 1 and incompleteness == 0:
        print("It seems that your conclusion doesn't match what we discussed. Try again, making sure to mention only the ideas we thought were good.")
    if incompleteness == 1 and incompleteness == 0:
        print("I think that you haven't mentioned all the ideas that we thought were good. Try again, please!")
    if incompleteness == 1 and incompleteness == 1:
        print("Mmm, I think that your conclusion is contradicting our discussion...and you forgot some ideas we thought were good! Try again :)")

#-----------------------------------------------------------

#Function to see if all topics have been covered

def Topics_covered(): #depends on New_topic_ask_opinion & EvaluateConclusion
    #I check the values stored in the entity dictionary and append them to separate sets to work more easily
    #As I am using sets, I can add the topics not covered over and over again and they won't be repeated (not efficient, but easy)
    for key, value in COVERED_TOPIC.items(): 
        if value == 'not_covered':
            topics_NOTcovered.add(key)
        if value == 'covered':
            topics_covered.add(key)
            if key in topics_NOTcovered:
                topics_NOTcovered.remove(key)
    #Some topic left
    if 'not_covered' in COVERED_TOPIC.values():
        New_topic_ask_opinion()
    #All topics are covered
    if 'not_covered' not in COVERED_TOPIC.values():
        print("Great, it seems that we've looked at all the options.")

#-----------------------------------------------------------

#Function to give feedback related to connectors

def Feedback_connectors(): #depends on ClassifyREASON & Repetitive_connector
    print("I'll give you some feedback on how you linked your ideas")
    #Feedback on variety of connectors
    Repetitive_connector(user_utterances)
    
    #Feedback on ammount of connectors
    overall_connectors = 0
    #I analyze each utterance and update the counter
    for utterance in user_utterances:
        overall_connectors += ClassifyREASON(utterance)
    print("I've made some calculations and I see that you've used " + str(overall_connectors) + " connectors.")
    if overall_connectors < 7:
        print("Those aren't many connectors :( Take a look at the list of connectors I'll show you")
    if overall_connectors > 7:
        print("Good job, those are quite a few! It seems that you know how to structure your ideas, but have a look at some advanced connectors that you might find useful:")
    #I display a list for the student to see new connectors
    list_of_connectors_to_study = open("Connectors.txt", encoding='utf-8')
    print(list_of_connectors_to_study.read())

    

##* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►
#-------------------------------------------------------------------------------------
##THE DISCUSSION PROMPT
discussion_PROBLEM = "Imagine that a town wants to attract more tourists. \
I will give you some ideas that the Town Council is considering, \
and you and your robot partner will have to discuss which of those ideas you think are good and why."
discussion_IDEA1 = "Building a large nightclub"
discussion_IDEA2 = "Putting up security cameras"
discussion_IDEA3 = "Having more shops"
discussion_IDEA4 = "Providing parks"
discussion_IDEA5 = "Building holiday flats"
#-------------------------------------------------------------------------------------
##GREETING THE USER AND EXPLAINING THE PROCESS
print("Hello! This is a tool to help you practice discussing in English.")
print("You will be given instructions and then your robot partner will be able to discuss with you.")
anykey = input("Press ENTER to continue")
print("\n")
print(discussion_PROBLEM)
print("\n")
print("These are the ideas:")
print(discussion_IDEA1)
print(discussion_IDEA2)
print(discussion_IDEA3)
print(discussion_IDEA4)
print(discussion_IDEA5)
print("\n")
anykey = input("Press ENTER to continue")


#* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►

#THE CHAT

#Starting the timer
start_time = time.time()

discussion_ended = 0
utterance = input("Hi! I'm your robot partner. You can start saying what you think about one of the ideas\n")
chat_hi_hasopinion(utterance)
chat_hasreason(utterance)


#This function checks if all topics are covered to get to the conclusion
#Here that won't be the case, so early, so it will run the function that presents a new topic
Topics_covered()

#The chat loop ends once all topics have been covered
while len(topics_covered) != 5:
    utterance = input("What do you think about that topic?\n")
    chat_hi_hasopinion(utterance)
    chat_hasreason(utterance)
    Topics_covered()

#The conclusion lopp ends once a valid conclusion is given
conclusion_valid = []
conclusion = input("So, what would you say are the ideas that we liked the most?\n")
EvaluateConclusion(conclusion)
while len(conclusion_valid) < 1:
    conclusion = input("Think again, what would you say are the ideas that we liked the most?\n")
    EvaluateConclusion(conclusion)

#Stopping the timer
duration = (time.time() - start_time)


#* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►

#FEEDBACK

print("▁ ▂ ▄ ▅ ▆ ▇ █ YOUR FEEDBACK █ ▇ ▆ ▅ ▄ ▂ ▁\n")

#Duration feedback
print("▁ ▂ ▄ Time Management ▄ ▂ ▁")
print('The discussion took ',duration,' seconds')
if duration > 300:
    print("It's good that you are able to carry a long discussion, but remember that in the FCE exam you have four minutes to discuss all the ideas! Practice summarizing a bit")
if duration < 300 and duration > 200:
    print("Seems like you are good at managing time ;)")
if duration < 250:
    print("You discussed the ideas too quickly! Next time, try giving more details (reasons, examples) to make stronger arguments")

#Connectors feedback
print("▁ ▂ ▄ Use of Connectors ▄ ▂ ▁")    
Feedback_connectors()
print("It's been a pleasure talking to you! (*＾▽＾)／")

#* ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► * ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ► ◄ ◊ ►
