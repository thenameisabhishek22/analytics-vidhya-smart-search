from sentence_transformers import SentenceTransformer, util
import torch
import gradio as gr

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

course_data = [
    {"title": "Introduction to Python", "description": "Power up your career with the best and most popular data science language, Python. Leverage your Python skills to start your Data Science journey. This course is intended for beginners with no coding or Data Science background."},
    {"title": "Introduction to AI & ML", "description": "Artificial Intelligence (AI) and Machine Learning (ML) are changing the world around us. From functions to industries, AI and ML are disrupting how we work and how we function. Get to know all about the different facets of AI and ML in this course."},
    {"title": "Machine Learning Certification Course for Beginners", "description": "In this free machine learning certification course, you will learn Python, the basics of machine learning, how to build machine learning models, and feature engineering techniques to improve the performance of your machine learning models."},
    {"title": "Data Preprocessing on a Real-World Problem Statement - Free Course", "description": "Kickstarting your career in Data Science can be easy with the right guide. This free course will serve as the perfect path to help beginners navigate the complex terrain of Data Preprocessing and prepare any data for modelling."},
    {"title": "Introduction to Business Analytics", "description": "Business analytics is thriving – and so is its role in forward-thinking organizations around the world. The demand for business analytics professionals is growing multifold - and now is the time to start working towards your desired career."},
    {"title": "Framework to Choose the Right LLM for your Business", "description": "This course provides a comprehensive framework for selecting the right LLM for your business. Learn to evaluate LLMs based on accuracy, cost, scalability, and more, while exploring real-world applications to make informed, strategic AI decisions."},
    {"title": "Improving Real World RAG Systems: Key Challenges & Practical Solutions", "description": "Master key challenges in real-world Retrieval-Augmented Generation (RAG) systems. Explore practical solutions, advanced retrieval strategies, and agentic RAG systems to improve context, relevance, and accuracy in AI-driven applications."},
    {"title": "Building Smarter LLMs with Mamba and State Space Model", "description": "Master Mamba's selective state space model for LLMs. Discover key components like the Mamba block, optimizing sequence modeling with efficient, scalable training and inference, surpassing traditional Transformers."},
    {"title": "GenAI Applied to Quantitative Finance: For Control Implementation", "description": "Embark on the journey to understand quantitative finance with GenAI. Learn to implement AI-driven control systems for trading, risk management, and predictive modeling, optimizing financial decision-making and performance."},
    {"title": "Navigating LLM Tradeoffs: Techniques for Speed, Cost, Scale & Accuracy", "description": "Master the art of optimizing LLMs with practical techniques to achieve the best balance of performance and cost."},
    {"title": "Generative AI - A Way of Life - Free Course", "description": "Embark on a journey into Generative AI for beginners. Learn AI-powered text and image generation, use top AI tools, and explore industry applications. Gain practical skills, understand ethical practices, and master prompting techniques."},
    {"title": "MidJourney: From Inspiration to Implementation - Free Course", "description": "Understand the fundamentals of the famous image generation tool - Midjourney in this free course. You will learn the various components of Midjourney and how to use it to bring your imaginations to real world."},
    {"title": "Exploring Stability.AI - Free Course", "description": "Explore Stability.AI with this free course providing hands-on experience. Learn to deploy SD WebUI, use Automatic WebUI on RunPod GPU environments, and master installation, setup, generation, and customization of SD."},
    {"title": "Microsoft Excel: Formulas & Functions", "description": "Microsoft Excel is still the tool of choice in the industry when it comes to performing data analysis, thanks to its incredible depth and array of formulas and functions. This course covers a wide range of Excel formulas, including LookUp Functions!"},
    {"title": "Tableau for Beginners", "description": "Tableau is the tool of choice for business intelligence, analytics and data visualization experts. Learn how to use Tableau, the different features of Tableau, and start building impactful visualization using this Tableau tutorial!"},
    {"title": "Twitter Sentiment Analysis", "description": "What is sentiment analysis? Why is sentiment analysis so popular in data science? And how can you perform sentiment analysis? Find the answers to all these questions in this free course on Sentiment Analysis using Python!"},
    {"title": "Time Series Forecasting using Python", "description": "Learn time series analysis and build your first time series forecasting model using ARIMA, Holt’s Winter and other time series forecasting methods in Python for a real-life industry use case"},
    {"title": "Loan Prediction Practice Problem (Using Python)", "description": "This course is aimed for people getting started into Data Science and Machine Learning while working on a real life practical problem."},
    {"title": "The A to Z of Unsupervised ML - Free Course", "description": "Get ahead of the crowd with this free course on Unsupervised Machine Learning Models. We will be covering popular clustering algorithms and DBSCAN and show you its applications on a real-world business problem."},
    {"title": "The Working of Neural Networks - Free Course", "description": "Kickstarting your career in the field of Deep Learning can be made easy with the right guide. This course will serve as a learning path to help beginners navigate through the complex terrain of Deep Learning."}
]

course_embeddings = model.encode([course['title'] + " " + course['description'] for course in course_data], convert_to_tensor=True)

print(f"Course Embeddings Shape: {course_embeddings.shape}")

def search_courses(query):
    try:
        if not query or query.strip() == "":
            return "Please enter a valid query."

        query_embedding = model.encode(query, convert_to_tensor=True)
        print(f"Query Embedding Shape: {query_embedding.shape}")  

        cosine_scores = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]
        print(f"Cosine Scores: {cosine_scores}")  

        
        if len(query.split()) <= 3:
            threshold = 0.3  
        else:
            threshold = 0.5  

        relevant_courses = [(idx, score) for idx, score in enumerate(cosine_scores) if score >= threshold]

        if len(relevant_courses) == 0:
            for idx, course in enumerate(course_data):
                if query.lower() in course['title'].lower() or query.lower() in course['description'].lower():
                    relevant_courses.append((idx, torch.tensor(1.0)))  

        relevant_courses = sorted(relevant_courses, key=lambda x: x[1], reverse=True)

        if len(relevant_courses) == 0:
            return "No relevant courses found for your query."

        results = ""
        for idx, score in relevant_courses:
            results += f"Title: {course_data[idx]['title']}\n"
            results += f"Description: {course_data[idx]['description']}\n"
            results += f"Similarity Score: {score:.2f}\n\n"  

        return results

    except Exception as e:
        error_message = f"Error occurred: {e}"
        print(error_message)
        return error_message


interface = gr.Interface(
    fn=search_courses,  
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),  
    outputs="text",  
    title="Smart Course Search",  
    description="Enter your query to search for relevant free courses on Analytics Vidhya."
)

interface.launch()