from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
text = '''
Artificial Intelligence, or AI, refers to computer systems designed to mimic human intelligence to perform complex tasks.

Machine learning is a core subset of AI where algorithms learn patterns from massive amounts of data without explicit programming.

Deep learning uses neural networks inspired by the human brain to process data like images, speech, and text.

Large Language Models (LLMs) are AI systems trained on vast text datasets to understand, generate, and translate human language.

Computer vision allows AI to interpret and analyze visual information from the world, such as identifying objects in photos.

Natural Language Processing (NLP) enables computers to talk, write, and understand human context and sentiment.

AI systems rely heavily on massive computational power and high-quality data to improve their accuracy over time.

Virtual assistants like Siri and Google Assistant use AI to understand voice commands and manage daily schedules.

Recommendation algorithms on platforms like Netflix and Spotify analyze your past behavior to predict what you will enjoy next.

Navigation apps use real-time AI processing to analyze traffic data and calculate the fastest routes.

AI-powered smart home devices optimize energy consumption by learning your daily habits and preferences.

In cybersecurity, AI monitors network traffic to detect and block potential threats or unusual activity instantly.

Generative AI tools can create entirely new text, code, images, and music based on simple user prompts.

Autonomous vehicles use AI, sensors, and cameras to navigate roads and make split-second driving decisions.

In healthcare, AI helps doctors analyze medical scans to detect diseases like cancer much earlier than traditional methods.

E-commerce businesses use AI chatbots to provide instant, round-the-clock customer service and support.

Automated manufacturing lines use AI robotics to perform precise, repetitive tasks with minimal human error.

AI is heavily used in finance to analyze stock market trends and execute high-speed algorithmic trading.

The integration of AI into the workforce is shifting jobs toward managing AI tools rather than performing manual data entry.

Ethical considerations, such as data privacy and algorithmic bias, are critical challenges as AI continues to integrate into daily life.
'''
chunks = text.split(".")
print(chunks)

chunks=[chunk.strip() for chunk in chunks if chunk.strip()]
print(chunks)


model = SentenceTransformer('all-MiniLM-L6-v2')
chunks=['Artificial Intelligence, or AI, refers to computer systems designed to mimic human intelligence to perform complex tasks', 'Machine learning is a core subset of AI where algorithms learn patterns from massive amounts of data without explicit programming', 'Deep learning uses neural networks inspired by the human brain to process data like images, speech, and text', 'Large Language Models (LLMs) are AI systems trained on vast text datasets to understand, generate, and translate human language', 'Computer vision allows AI to interpret and analyze visual information from the world, such as identifying objects in photos', 'Natural Language Processing (NLP) enables computers to talk, write, and understand human context and sentiment', 'AI systems rely heavily on massive computational power and high-quality data to improve their accuracy over time', 'Virtual assistants like Siri and Google Assistant use AI to understand voice commands and manage daily schedules', 'Recommendation algorithms on platforms like Netflix and Spotify analyze your past behavior to predict what you will enjoy next', 'Navigation apps use real-time AI processing to analyze traffic data and calculate the fastest routes', 'AI-powered smart home devices optimize energy consumption by learning your daily habits and preferences', 'In cybersecurity, AI monitors network traffic to detect and block potential threats or unusual activity instantly', 'Generative AI tools can create entirely new text, code, images, and music based on simple user prompts', 'Autonomous vehicles use AI, sensors, and cameras to navigate roads and make split-second driving decisions', 'In healthcare, AI helps doctors analyze medical scans to detect diseases like cancer much earlier than traditional methods', 'E-commerce businesses use AI chatbots to provide instant, round-the-clock customer service and support', 'Automated manufacturing lines use AI robotics to perform precise, repetitive tasks with minimal human error', 'AI is heavily used in finance to analyze stock market trends and execute high-speed algorithmic trading', 'The integration of AI into the workforce is shifting jobs toward managing AI tools rather than performing manual data entry', 'Ethical considerations, such as data privacy and algorithmic bias, are critical challenges as AI continues to integrate into daily life']

embedding=model.encode(chunks)
print(embedding.shape)


#document chunks
chunks=['Artificial Intelligence, or AI, refers to computer systems designed to mimic human intelligence to perform complex tasks', 'Machine learning is a core subset of AI where algorithms learn patterns from massive amounts of data without explicit programming', 'Deep learning uses neural networks inspired by the human brain to process data like images, speech, and text', 'Large Language Models (LLMs) are AI systems trained on vast text datasets to understand, generate, and translate human language', 'Computer vision allows AI to interpret and analyze visual information from the world, such as identifying objects in photos', 'Natural Language Processing (NLP) enables computers to talk, write, and understand human context and sentiment', 'AI systems rely heavily on massive computational power and high-quality data to improve their accuracy over time', 'Virtual assistants like Siri and Google Assistant use AI to understand voice commands and manage daily schedules', 'Recommendation algorithms on platforms like Netflix and Spotify analyze your past behavior to predict what you will enjoy next', 'Navigation apps use real-time AI processing to analyze traffic data and calculate the fastest routes', 'AI-powered smart home devices optimize energy consumption by learning your daily habits and preferences', 'In cybersecurity, AI monitors network traffic to detect and block potential threats or unusual activity instantly', 'Generative AI tools can create entirely new text, code, images, and music based on simple user prompts', 'Autonomous vehicles use AI, sensors, and cameras to navigate roads and make split-second driving decisions', 'In healthcare, AI helps doctors analyze medical scans to detect diseases like cancer much earlier than traditional methods', 'E-commerce businesses use AI chatbots to provide instant, round-the-clock customer service and support', 'Automated manufacturing lines use AI robotics to perform precise, repetitive tasks with minimal human error', 'AI is heavily used in finance to analyze stock market trends and execute high-speed algorithmic trading', 'The integration of AI into the workforce is shifting jobs toward managing AI tools rather than performing manual data entry', 'Ethical considerations, such as data privacy and algorithmic bias, are critical challenges as AI continues to integrate into daily life']

model = SentenceTransformer('all-MiniLM-L6-v2')
chunks=['Artificial Intelligence, or AI, refers to computer systems designed to mimic human intelligence to perform complex tasks', 'Machine learning is a core subset of AI where algorithms learn patterns from massive amounts of data without explicit programming', 'Deep learning uses neural networks inspired by the human brain to process data like images, speech, and text', 'Large Language Models (LLMs) are AI systems trained on vast text datasets to understand, generate, and translate human language', 'Computer vision allows AI to interpret and analyze visual information from the world, such as identifying objects in photos', 'Natural Language Processing (NLP) enables computers to talk, write, and understand human context and sentiment', 'AI systems rely heavily on massive computational power and high-quality data to improve their accuracy over time', 'Virtual assistants like Siri and Google Assistant use AI to understand voice commands and manage daily schedules', 'Recommendation algorithms on platforms like Netflix and Spotify analyze your past behavior to predict what you will enjoy next', 'Navigation apps use real-time AI processing to analyze traffic data and calculate the fastest routes', 'AI-powered smart home devices optimize energy consumption by learning your daily habits and preferences', 'In cybersecurity, AI monitors network traffic to detect and block potential threats or unusual activity instantly', 'Generative AI tools can create entirely new text, code, images, and music based on simple user prompts', 'Autonomous vehicles use AI, sensors, and cameras to navigate roads and make split-second driving decisions', 'In healthcare, AI helps doctors analyze medical scans to detect diseases like cancer much earlier than traditional methods', 'E-commerce businesses use AI chatbots to provide instant, round-the-clock customer service and support', 'Automated manufacturing lines use AI robotics to perform precise, repetitive tasks with minimal human error', 'AI is heavily used in finance to analyze stock market trends and execute high-speed algorithmic trading', 'The integration of AI into the workforce is shifting jobs toward managing AI tools rather than performing manual data entry', 'Ethical considerations, such as data privacy and algorithmic bias, are critical challenges as AI continues to integrate into daily life']

embedding=model.encode(chunks)
print(embedding.shape)
query=input("enter query:")
query_embedding=model.encode(query)

similarities = cosine_similarity(
    [query_embedding],
    embedding
)

best_match_index=similarities.argmax()
print(chunks[best_match_index])
