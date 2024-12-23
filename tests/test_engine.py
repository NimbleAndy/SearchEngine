from microsearch.engine import SearchEngine
from math import log
import logging

logging.basicConfig(level=logging.INFO)


"""
Conceptual understanding I used this "https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k#precision-example"
Code implementation I used this: "https://www.pinecone.io/learn/offline-evaluation/"
#I used 10, the examples I've seen were 3, no biggy.
"""

# Precision at K@10
def precision_at_k(ranked_list, relevant_docs, k=10):
    retrieved_docs = ranked_list[:k]
    relevant_count = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    return relevant_count / k


# Recall at K@10
def recall_at_k(ranked_list, relevant_docs, k=10):
    retrieved_docs = ranked_list[:k]
    relevant_count = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    return relevant_count / len(relevant_docs) if relevant_docs else 0


# Mean Reciprocal Rank
def reciprocal_rank(ranked_list, relevant_docs):
    for i, doc in enumerate(ranked_list, start=1):
        if doc in relevant_docs:
            return 1 / i
    return 0


# Normalized Discounted Cumulative Gain from lecture!! Same concept I used in the lab.
def ndcg(ranked_list, relevant_docs, k=10):
    def dcg(ranked_list, relevant_docs, k):
        return sum(1 / log(i + 2, 2) if ranked_list[i] in relevant_docs else 0 for i in range(min(k, len(ranked_list))))

    ideal_list = relevant_docs[:k]
    ideal_dcg = dcg(ideal_list, relevant_docs, k)
    actual_dcg = dcg(ranked_list, relevant_docs, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0 #Avoid division by zero!


# Evaluate the engine with multiple queries
def evaluate_engine(engine, test_queries):
    metrics = {"precision@10": [], "recall@10": [], "mrr": [], "ndcg": []} ##This is the final result of the IR performance

    for query, relevant_docs in test_queries.items():
        results = engine.search(query)  # Returns a dict {docURL: score}
        ranked_docs = [doc for doc, score in sorted(results.items(), key=lambda x: x[1], reverse=True)]
        print(f"Query: {query}")
        print(f"Ranked Docs: {ranked_docs}")
        # Compute metrics....
        p10 = precision_at_k(ranked_docs, relevant_docs, k=10)
        r10 = recall_at_k(ranked_docs, relevant_docs, k=10)
        mrr = reciprocal_rank(ranked_docs, relevant_docs)
        ndcg_score = ndcg(ranked_docs, relevant_docs, k=10)

        # we save it here!
        metrics["precision@10"].append(p10)
        metrics["recall@10"].append(r10)
        metrics["mrr"].append(mrr)
        metrics["ndcg"].append(ndcg_score)

        # Calculate averages
    for metric, scores in metrics.items():
        metrics[metric] = sum(scores) / len(scores) if scores else 0

    return metrics


# Test the SearchEngine class
def test_search_engine():
    engine = SearchEngine()

    logging.info("Indexing TESTING documents...")
    documents = [
        ("doc1", "Python is a popular programming language."),
        ("doc2", "Machine learning enables AI applications."),
        ("doc3", "Python is great for data analysis."),
        ("doc4", "Hello World! This is my first program!"),
        ("doc5", "The sun sets softly over the rolling hills, as I hold you close and our hearts instill a love that "
                 "time can never still"),
        ("doc6", "Through the chaos of this world, your guiding light shines bright, leading me home to the safety of "
                 "your arms tonight."),
        ("doc7", "Beneath the starry sky, our hands entwined, a symphony of souls in perfect harmony, "
                 "forever intertwined."),
        ("doc8",
         "Python is a versatile programming language that is widely used in data science, machine learning, "
         "and web development."),
        ("doc9",
         "Machine learning algorithms can improve the efficiency of software systems by analyzing data patterns."),
        ("doc10",
         "The history of space exploration includes milestones like the Apollo moon landing and Mars rover missions."),
        ("doc11", "Cooking techniques such as sous-vide and fermentation have gained popularity among modern chefs."),
        ("doc12", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union."),
        ("doc13",
         "Renewable energy sources like solar and wind power are becoming increasingly important for sustainability."),
        ("doc14", "The works of Shakespeare, including Hamlet and Macbeth, are considered timeless literary classics."),
        ("doc15", "The Great Barrier Reef in Australia is home to an incredible diversity of marine life."),
        ("doc16",
         "Advances in genetics have enabled breakthroughs in personalized medicine and gene editing technologies."),
        (
            "doc17",
            "Artificial intelligence and robotics are transforming industries like healthcare and manufacturing."),
        ("doc18", "The Renaissance was a cultural movement that profoundly influenced art, science, and literature."),
        ("doc19",
         "Cryptocurrency and blockchain technology are reshaping the financial landscape with decentralized systems."),
        (
            "doc20",
            "The biodiversity crisis threatens numerous species, highlighting the need for conservation efforts."),
        (
            "doc21",
            "Quantum computing has the potential to revolutionize fields like cryptography and material science."),
        ("doc22", "The Internet of Things (IoT) connects devices, enabling smarter homes and cities."),
        ("doc23",
         "Space exploration has led to numerous technological advancements, from satellite communications to GPS technology."),
        ("doc24",
         "Renewable energy sources like solar, wind, and hydroelectric power are essential for combating climate change and reducing carbon emissions."),
        ("doc25",
         "Advances in medicine, including gene editing and personalized therapies, are transforming the way we treat diseases."),
        ("doc26",
         "Artificial intelligence is reshaping industries such as healthcare, transportation, and finance by automating processes and uncovering insights."),
        ("doc27",
         "Ocean conservation efforts are crucial to preserving marine ecosystems and ensuring sustainable fishing practices."),
        ("doc28",
         "The advent of 5G technology promises faster internet speeds and more reliable connections, enabling new possibilities for businesses and consumers."),
        ("doc29",
         "Cybersecurity remains a critical concern as data breaches and cyberattacks grow in frequency and sophistication."),
        ("doc30",
         "Climate change is causing more frequent extreme weather events, including hurricanes, wildfires, and droughts, impacting communities worldwide."),
        ("doc31",
         "Cryptocurrencies, like Bitcoin and Ethereum, are disrupting traditional finance and raising questions about regulation and sustainability."),
        ("doc32",
         "Electric vehicles are reducing greenhouse gas emissions and dependency on fossil fuels, but battery production raises environmental concerns."),
        ("doc33",
         "The COVID-19 pandemic highlighted the importance of global health systems and rapid vaccine development to mitigate crises."),
        ("doc34",
         "Machine learning techniques, such as neural networks and reinforcement learning, are being used to solve complex problems."),
        ("doc35",
         "Space tourism is becoming a reality, with private companies offering opportunities to travel beyond Earth’s atmosphere."),
        ("doc36",
         "The study of dark matter and dark energy is expanding our understanding of the universe’s composition and evolution."),
        ("doc37",
         "Urban planning must adapt to accommodate population growth, climate resilience, and the integration of green spaces."),
        ("doc38",
         "Social media platforms have transformed communication, influencing politics, culture, and individual mental health."),
        ("doc39",
         "Wearable technology, including fitness trackers and smartwatches, is empowering individuals to monitor their health and wellbeing."),
        ("doc40",
         "Robotics advancements are creating opportunities in manufacturing, space exploration, and even caregiving for the elderly."),
        ("doc41",
         "Gene editing technologies like CRISPR are revolutionizing genetics and opening doors to curing hereditary diseases."),
        ("doc42",
         "The role of renewable energy in achieving net-zero carbon emissions is central to international climate agreements."),
        ("doc43",
         "3D printing is revolutionizing manufacturing, allowing for rapid prototyping, customization, and even the creation of human tissue."),
        ("doc44",
         "The rise of telemedicine during the pandemic has shown the potential of remote healthcare delivery and accessibility."),
        ("doc45",
         "Data science and big data analytics are transforming industries by enabling predictive insights and data-driven decision-making."),
        ("doc46",
         "Virtual reality (VR) and augmented reality (AR) are enhancing gaming, education, and even medical training."),
        ("doc47",
         "Supply chain disruptions, driven by geopolitical tensions and the pandemic, highlight the importance of diversification and resilience."),
        ("doc48",
         "The exploration of Mars and the search for extraterrestrial life are advancing planetary science and space technology."),
        ("doc49",
         "Blockchain technology is being used beyond cryptocurrencies to create secure, transparent, and decentralized systems."),
        ("doc50",
         "Renewable agriculture practices, such as crop rotation and agroforestry, aim to create sustainable food systems."),
        ("doc51",
         "Autonomous vehicles are transforming transportation, reducing accidents, and optimizing traffic flow."),
        ("doc52",
         "Artificial intelligence is playing a crucial role in drug discovery, accelerating research and reducing costs."),
        ("doc53",
         "Renewable energy storage solutions, such as advanced batteries and hydrogen fuel cells, are critical for energy transition."),
        ("doc54",
         "Cloud computing provides scalable infrastructure for businesses, enabling remote work and big data "
         "processing."),
        ("doc55",
         "The integration of IoT and smart grids is improving energy efficiency and reliability in power distribution "
         "systems."),
        ("doc56",
         "E-commerce platforms are reshaping retail, offering personalized shopping experiences and faster delivery."),
        ("doc57", "Augmented reality applications are enhancing education, marketing, and industrial training."),
        ("doc58", "The global food crisis is driving innovation in vertical farming and lab-grown meat production."),
        (
            "doc59",
            "Quantum cryptography offers unbreakable security by leveraging the principles of quantum mechanics."),
        ("doc60",
         "Biodiversity loss is accelerating due to habitat destruction, climate change, and pollution, impacting "
         "ecosystems."),
        ("doc61",
         "AI-powered chatbots are revolutionizing customer service, providing faster and more accurate responses."),
        ("doc62",
         "Renewable energy integration into national grids requires innovative technologies and policy frameworks."),
        ("doc63",
         "The rise of e-learning platforms is democratizing education, making quality learning accessible to all."),
        ("doc64", "Space debris mitigation strategies are essential for the sustainability of orbital activities."),
        ("doc65",
         "The rise of social entrepreneurship is addressing global challenges through innovative business models."),
        ("doc66", "The ethics of AI development are raising concerns about bias, transparency, and accountability."),
        ("doc67", "Global deforestation is a major driver of biodiversity loss and climate change."),
        ("doc68", "The role of nanotechnology in medicine is advancing drug delivery systems and diagnostics."),
        ("doc69", "The evolution of renewable energy technologies is reducing dependence on fossil fuels."),
        ("doc70", "Digital health technologies are empowering patients to take control of their own health."),
    ]
    engine.bulk_index(documents)

    # Define test queries
    test_queries = {
        "programming": ["doc1", "doc3"],
        "AI applications": ["doc2"],
        "python programming": ["doc1", "doc3"],
        "python": ["doc1", "doc3", "doc8"],
        "nonexistent term": [],
        "machine": ["doc2"],
        "language": ["doc1", "doc2"],
        "our hands in by Coldplay": ["doc7"],
        "Eternal Embrace by Jane Doe": ["doc5"],
        "Finding My Way Back to You by Michael Bublé": ["doc6"],
        "what should be my first program in python": ["doc4"],
        "benefits of machine learning in software systems": ["doc9", "doc17"],
        "techniques for modern cooking and food preservation": ["doc11"],
        "exploration milestones in space history": ["doc10"],
        "the impact of renewable energy on sustainability": ["doc13"],
        "analyzing themes in Shakespeare's Hamlet and Macbeth": ["doc14"],
        "marine biodiversity in the Great Barrier Reef": ["doc15"],
        "advances in genetics and personalized medicine": ["doc16"],
        "the Cold War and its geopolitical implications": ["doc12"],
        "art and science during the Renaissance": ["doc18"],
        "applications of blockchain and cryptocurrency": ["doc19"],
        "biodiversity crisis and conservation efforts": ["doc20"],
        "future potential of quantum computing": ["doc21"],
        "smart homes enabled by IoT technology": ["doc22"],
        "renewable energy versus fossil fuels": ["doc13"],
        "geopolitical tensions during the Cold War": ["doc12"],
        "advancements in space technology": ["doc23", "doc48"],
        "renewable energy sources and climate change": ["doc24", "doc42"],
        "artificial intelligence applications in healthcare": ["doc26", "doc34"],
        "ocean conservation and sustainable fishing": ["doc27"],
        "cryptocurrency and blockchain technology": ["doc31", "doc49"],
        "electric vehicles and battery production": ["doc32"],
        "3D printing in healthcare and manufacturing": ["doc43"],
        "telemedicine and remote healthcare delivery": ["doc44"],
        "robotics in manufacturing and caregiving": ["doc40"],
        "gene editing and CRISPR technology": ["doc41"],
        "renewable agriculture and sustainable food systems": ["doc50"],
        "big data analytics in industries": ["doc45"],
        "climate change and extreme weather events": ["doc30"],
        "dark matter and the evolution of the universe": ["doc36"],
        "urban planning and green spaces": ["doc37"],
        "autonomous vehicles and traffic optimization": ["doc51", "doc54"],
        "AI in drug discovery and healthcare": ["doc52", "doc61"],
        "renewable energy storage solutions": ["doc53", "doc69"],
        "cloud computing and big data processing": ["doc54", "doc56"],
        "IoT and smart grid integration": ["doc55", "doc62"],
        "e-commerce and personalized shopping experiences": ["doc56", "doc63"],
        "augmented reality in education and marketing": ["doc57", "doc63"],
        "biodiversity loss and climate change": ["doc60", "doc67"],
        "quantum cryptography and security": ["doc59", "doc48"],
        "food crisis and lab-grown meat": ["doc58", "doc50"],
        "space debris mitigation strategies": ["doc64", "doc36"],
        "social entrepreneurship and global challenges": ["doc65", "doc38"],
        "AI ethics and bias concerns": ["doc66", "doc52"],
        "nanotechnology in medicine": ["doc68", "doc41"],
        "digital health technologies and patient empowerment": ["doc70", "doc44"],
        "deforestation and biodiversity loss": ["doc67", "doc60"],
        "renewable energy integration and storage": ["doc62", "doc53", "doc42"],
        "vertical farming and food sustainability": ["doc58", "doc50"],
        "space exploration and orbital sustainability": ["doc64", "doc48"],
        "AI-powered chatbots and customer service": ["doc61", "doc52"],
    }

    if engine._embedding_matrix is None:
        logging.warning("Embeddings are not built. Search results might not work properly.")
    else:
        logging.info("Embeddings are ready.")

    logging.info("Evaluating search engine...")
    metrics = evaluate_engine(engine, test_queries)
    logging.info("Evaluation completed.")

    print("\nEvaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    test_search_engine()