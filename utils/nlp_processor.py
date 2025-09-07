"""
Enhanced Natural Language Processing module for understanding user queries.
Handles intent recognition, entity extraction, and query understanding.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import logging

from config import (
    HUGGINGFACE_MODEL, MAX_INPUT_LENGTH, 
    EMBEDDING_DIMENSION, SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class NLPProcessor:
    """Processes natural language queries to extract crop and symptom information."""
    
    def __init__(self):
        # Initialize tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
        self.model = AutoModel.from_pretrained(HUGGINGFACE_MODEL)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.classifier = None
        self.faiss_index = None
        self.symptom_texts = []
        
    def train_intent_classifier(self, training_data):
        """
        Train a simple intent classifier to distinguish between:
        - Disease diagnosis requests
        - Cultivation advice requests
        - Region compatibility queries
        
        Parameters:
            training_data (list): List of (text, label) tuples
        """
        texts, labels = zip(*training_data)
        self.classifier = make_pipeline(
            self.vectorizer, 
            MultinomialNB()
        )
        self.classifier.fit(texts, labels)
    
    def get_embedding(self, text):
        """
        Generate embedding for text using Hugging Face model.
        
        Parameters:
            text (str): Input text to embed
            
        Returns:
            numpy.array: Text embedding
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_INPUT_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling to get sentence embedding
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        return embedding.numpy()
    
    def build_faiss_index(self, symptom_descriptions):
        """
        Build FAISS index for symptom similarity search.
        
        Parameters:
            symptom_descriptions (list): List of symptom descriptions
        """
        self.symptom_texts = symptom_descriptions
        embeddings = []
        
        for desc in symptom_descriptions:
            embedding = self.get_embedding(desc)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.faiss_index.add(embeddings)
    
    def find_similar_symptoms(self, query, k=5):
        """
        Find similar symptoms using FAISS index.
        
        Parameters:
            query (str): User query about symptoms
            k (int): Number of similar symptoms to return
            
        Returns:
            list: Indices of similar symptoms
        """
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Filter results by similarity threshold
        similar_indices = []
        for i, distance in zip(indices[0], distances[0]):
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            if similarity >= SIMILARITY_THRESHOLD:
                similar_indices.append(i)
        
        return similar_indices
    
    def extract_entities(self, text):
        """
        Extract crop name and symptoms from user query using rule-based and ML approaches.
        
        Parameters:
            text (str): User query text
            
        Returns:
            dict: Extracted entities (crop, symptoms, etc.)
        """
        # Convert to lowercase for easier processing
        text_lower = text.lower()
        
        # Initialize entities
        entities = {
            'crop': None,
            'symptoms': [],
            'region': None,
            'soil_type': None,
            'intent': None
        }
        
        # Expanded Indian crop names with common synonyms
        crop_patterns = {
            'tomato': r'\b(tomato|tomatoes|tamatar)\b',
            'rice': r'\b(rice|paddy|dhan|chawal)\b',
            'wheat': r'\b(wheat|gehun|kanak)\b',
            'maize': r'\b(maize|corn|makka|bhutta)\b',
            'ragi': r'\b(ragi|finger millet|nachni)\b',
            'cotton': r'\b(cotton|kapas)\b',
            'sugarcane': r'\b(sugarcane|ganna|oob)\b',
            'coffee': r'\b(coffee|kapi)\b',
            'tea': r'\b(tea|chai)\b',
            'mango': r'\b(mango|mangoes|aam)\b',
            'banana': r'\b(banana|bananas|kela|plantain)\b',
            'grape': r'\b(grape|grapes|draksha|angoor)\b',
            'orange': r'\b(orange|oranges|narangi|santra)\b',
            'chili': r'\b(chili|chilli|mirchi)\b',
            'coconut': r'\b(coconut|nariyal|thengai)\b',
            'groundnut': r'\b(groundnut|peanut|moongphali)\b',
            'mustard': r'\b(mustard|sarson|rai)\b',
            'brinjal': r'\b(brinjal|eggplant|baingan)\b',
            'cabbage': r'\b(cabbage|patta gobi|bandh gobi)\b',
            'paddy': r'\b(paddy|dhan|rice seedling)\b',
            'potato': r'\b(potato|potatoes|aloo|batata)\b',
            'onion': r'\b(onion|onions|pyaz|ulli)\b',
            'turmeric': r'\b(turmeric|haldi)\b',
            'pepper': r'\b(pepper|kali mirch)\b',
            'cardamom': r'\b(cardamom|elaichi)\b',
            'arecanut': r'\b(arecanut|betel nut|supari)\b',
            'vegetables': r'\b(vegetables|sabzi|sag)\b',
            'flowers': r'\b(flowers|phool|pushp)\b',
            'spices': r'\b(spices|masala)\b',
            'pulses': r'\b(pulses|dal|legumes)\b'
        }
        
        # Compound symptom patterns (check these first)
        compound_symptom_patterns = {
            'yellow leaves': r'\b(yellow leaves|yellowing leaves|pila patte|pili pattiyan)\b',
            'brown spots': r'\b(brown spots|brown patches|bhure dhabbe|chitte)\b',
            'leaf curl': r'\b(leaf curl|curling leaves|murte hue patte|patton ka murjana)\b',
            'powdery mildew': r'\b(powdery mildew|white powder|safed chhaal|safed daag)\b',
            'bollworm': r'\b(bollworm|pod borer|guddi keed|tana keed)\b',
            'stem rot': r'\b(stem rot|stem decay|tana sadna|tana galna)\b',
            'root rot': r'\b(root rot|root decay|jadh sadna|jadh galna)\b',
            'fruit rot': r'\b(fruit rot|fruit decay|phal sadna|phal galna)\b',
            'wilting plants': r'\b(wilting plants|wilted plants|murjhana|sukna)\b',
            'black spots': r'\b(black spots|kale dhabbe|kale chitte)\b',
            'white spots': r'\b(white spots|safed dhabbe|safed chitte)\b',
            'leaf blight': r'\b(leaf blight|patton ka jhulsa|patton ka marjana)\b',
            'fruit drop': r'\b(fruit drop|phal girna|phal jharna)\b',
            'flower drop': r'\b(flower drop|phool girna|phool jharna)\b',
            'shoot borer': r'\b(shoot borer|kali keed|tana keed)\b',
            'aphid attack': r'\b(aphid attack|chosa|chidya)\b',
            'whitefly attack': r'\b(whitefly attack|safed makhi|safed patri)\b',
            'caterpillar attack': r'\b(caterpillar attack|sundhi|larva)\b',
            'mite attack': r'\b(mite attack|chhote keed|micro keed)\b',
            'nematode attack': r'\b(nematode attack|jadh keed|soil keed)\b'
        }
        
        # Base symptom patterns (check these after compound patterns)
        base_symptom_patterns = {
            'yellow': r'\b(yellow|yellowing|pila|pili)\b',
            'brown': r'\b(brown|browning|bhoora)\b',
            'spots': r'\b(spots|spotting|dhabbe|chitte)\b',
            'wilting': r'\b(wilting|wilt|murjhana|sukna)\b',
            'rot': r'\b(rot|rotting|sadna|galna)\b',
            'blight': r'\b(blight|jhulsa)\b',
            'mold': r'\b(mold|mould|fungus|fuljhari)\b',
            'holes': r'\b(holes|borer|ched|suran)\b',
            'curling': r'\b(curling|curl|murana)\b',
            'stunted': r'\b(stunted|stunt|rukna)\b',
            'dropping': r'\b(dropping|falling|girna)\b',
            'drying': r'\b(drying|dry|sukna)\b',
            'rust': r'\b(rust|jangal)\b',
            'powdery': r'\b(powdery|powder|safed chhaal)\b',
            'mildew': r'\b(mildew|fuljhari)\b',
            'aphid': r'\b(aphid|aphids|chosa)\b',
            'whitefly': r'\b(whitefly|safed makhi)\b',
            'caterpillar': r'\b(caterpillar|larva|sundhi)\b',
            'discoloration': r'\b(discoloration|color change|rang badalna)\b',
            'lesions': r'\b(lesions|sores|chhaal)\b',
            'boreholes': r'\b(boreholes|borer holes|keedon ke ched)\b',
            'larvae': r'\b(larvae|larva|sundhiyan)\b',
            'veins': r'\b(veins|veining|nasiyan)\b',
            'tips': r'\b(tips|ends|sirah)\b',
            'edges': r'\b(edges|margins|kinare)\b'
        }
        
        # Indian states and districts with better matching
        region_patterns = {
            'karnataka': r'\b(karnataka|karnatic|bangalore|bengaluru|mysore|mandya|tumkur|hassan|chikmagalur|kodagu|shivamogga|dharwad|belagavi|vijayapura|raichur|kolar|chickballapur)\b',
            'maharashtra': r'\b(maharashtra|mumbai|pune|nagpur|nashik|kolhapur|sangli|yavatmal|wardha)\b',
            'tamil nadu': r'\b(tamil nadu|tamilnadu|chennai|madras|coimbatore|thanjavur|nagapattinam)\b',
            'andhra pradesh': r'\b(andhra pradesh|andhra|hyderabad|vizag|guntur|prakasam|chittoor|anantapur)\b',
            'kerala': r'\b(kerala|trivandrum|kochi|calicut|idukki|wayanad|kottayam|pathanamthitta)\b',
            'punjab': r'\b(punjab|chandigarh|amritsar|ludhiana|jalandhar)\b',
            'haryana': r'\b(haryana|chandigarh|faridabad|gurgaon|hisar)\b',
            'uttar pradesh': r'\b(uttar pradesh|up|lucknow|kanpur|varanasi|allahabad)\b',
            'west bengal': r'\b(west bengal|w Bengal|kolkata|calcutta|darjeeling|dooars)\b',
            'gujarat': r'\b(gujarat|ahmedabad|vadodara|surat|rajkot|kutch)\b',
            'north india': r'\b(north india|northern india|north)\b',
            'south india': r'\b(south india|southern india|south)\b',
            'east india': r'\b(east india|eastern india|east)\b',
            'west india': r'\b(west india|western india|west)\b',
            'coastal': r'\b(coastal|coast|seaside|samundri)\b',
            'hills': r'\b(hills|hilly|mountain|pahadi|pahad)\b',
            'plains': r'\b(plains|plain|flatland|maidan)\b',
            'dry areas': r'\b(dry areas|dry region|sukha kshetra)\b',
            'rainfed': r'\b(rainfed|rain fed|barani)\b',
            'irrigated': r'\b(irrigated|irrigation|sinchit)\b'
        }
        
        # Soil type keywords with pattern matching
        soil_patterns = {
            'black soil': r'\b(black soil|black cotton soil|regur|kali mitti)\b',
            'red soil': r'\b(red soil|red earth|lal mitti)\b',
            'alluvial': r'\b(alluvial|alluvium|khadar|jari mitti)\b',
            'laterite': r'\b(laterite|lateritic)\b',
            'clay': r'\b(clay|clayey|chikni mitti)\b',
            'sandy': r'\b(sandy|sand|retili mitti|baloo mitti)\b',
            'loamy': r'\b(loamy|loam|dumat mitti)\b',
            'well-drained': r'\b(well[-\s]?drained|good drainage|drainage|nikasi)\b',
            'fertile': r'\b(fertile|rich soil|productive|upjau)\b',
            'acidic': r'\b(acidic|acid soil|low ph|amla)\b',
            'alkaline': r'\b(alkaline|alkali soil|high ph|kshariya)\b',
            'sandy loam': r'\b(sandy loam|sandy soil|retili dumat)\b',
            'clay loam': r'\b(clay loam|clayey loam|chikni dumat)\b',
            'silt': r'\b(silt|silty|silt soil)\b'
        }
        
        # Check for crop mentions using pattern matching
        for crop, pattern in crop_patterns.items():
            if re.search(pattern, text_lower):
                entities['crop'] = crop
                break
        
        # Check for compound symptom mentions first
        for symptom, pattern in compound_symptom_patterns.items():
            if re.search(pattern, text_lower):
                entities['symptoms'].append(symptom)
        
        # Then check for base symptom mentions (avoid duplicates)
        for symptom, pattern in base_symptom_patterns.items():
            if (re.search(pattern, text_lower) and 
                symptom not in ' '.join(entities['symptoms'])):
                entities['symptoms'].append(symptom)
        
        # Check for region mentions using pattern matching
        for region, pattern in region_patterns.items():
            if re.search(pattern, text_lower):
                entities['region'] = region
                break
        
        # Check for soil type mentions using pattern matching
        for soil, pattern in soil_patterns.items():
            if re.search(pattern, text_lower):
                entities['soil_type'] = soil
                break
        
        # Enhanced intent detection with better keyword matching
        cultivation_keywords = [
            r'\b(how to|how do i|how can i)\b.*\b(grow|cultivate|plant|farm|nurture)\b',
            r'\b(grow|cultivate|plant|farm)\b.*\b(method|technique|process|way|tips)\b',
            r'\b(cultivation|farming|growing)\b.*\b(tips|advice|guide|methods|practices)\b',
            r'\b(best practice|good practice|organic method|natural way)\b.*\b(for|to)\b',
            r'\b(organic|natural|traditional)\b.*\b(method|way|farming|cultivation)\b',
            r'\b(water requirement|irrigation need|watering schedule)\b',
            r'\b(fertilizer|manure|compost|nutrient)\b.*\b(need|require|application)\b'
        ]
        
        region_keywords = [
            r'\b(what|which)\b.*\b(crop|crops)\b.*\b(grow|suitable|good|best)\b.*\b(in|for|at)\b',
            r'\b(crop|crops)\b.*\b(for|in)\b.*\b(region|area|state|district|zone)\b',
            r'\b(suitable|compatible|appropriate|ideal)\b.*\b(crop|crops)\b',
            r'\b(best|ideal|good|recommended)\b.*\b(crop|crops)\b.*\b(for|in)\b',
            r'\b(region|area|state|district)\b.*\b(crop|crops|agriculture|farming)\b',
            r'\b(grow well|thrive|suitable|adapt)\b.*\b(in|to)\b'
        ]
        
        disease_keywords = [
            r'\b(treatment|remedy|solution|control|management)\b.*\b(for|of)\b',
            r'\b(how to|how do i)\b.*\b(treat|control|prevent|manage|handle)\b',
            r'\b(problem|issue|disease|pest|infection|attack)\b.*\b(with|in|on)\b',
            r'\b(symptom|sign|indication|evidence)\b.*\b(of|for)\b',
            r'\b(what is|what are)\b.*\b(this|these|the)\b.*\b(spots|wilting|yellowing|rot)\b',
            r'\b(identify|diagnose|recognize)\b.*\b(disease|problem|issue)\b',
            r'\b(cure|heal|fix|solve)\b.*\b(problem|disease|issue)\b'
        ]
        
        # Check for intent using pattern matching
        has_cultivation_intent = any(re.search(pattern, text_lower) for pattern in cultivation_keywords)
        has_region_intent = any(re.search(pattern, text_lower) for pattern in region_keywords)
        has_disease_intent = any(re.search(pattern, text_lower) for pattern in disease_keywords)
        
        # Use safe access
        has_symptoms = len(entities.get('symptoms', [])) > 0
        has_crop = entities.get('crop') is not None
        has_region = entities.get('region') is not None
        
        # Enhanced intent detection logic with priority for symptoms
        if has_disease_intent or has_symptoms:
            # If symptoms are mentioned, prioritize disease diagnosis
            entities['intent'] = 'disease_diagnosis'
        elif has_region_intent or has_region:
            # Region queries should be region compatibility
            entities['intent'] = 'region_compatibility'
        elif has_cultivation_intent or (has_crop and not has_symptoms):
            # Cultivation advice for crop-specific queries without symptoms
            entities['intent'] = 'cultivation_advice'
        else:
            # Default to unknown if we can't determine
            entities['intent'] = 'unknown'
        
        return entities
    
    def _extract_region(self, text):
        """Helper method to extract region from text (simplified)."""
        # This would be enhanced with proper NER or geocoding
        region_keywords = ['north', 'south', 'east', 'west', 'coastal', 'mountain']
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if word in region_keywords and i+1 < len(words):
                return f"{word} {words[i+1]}"
        
        return "unknown"
    
    def _extract_soil_type(self, text):
        """Helper method to extract soil type from text."""
        soil_keywords = {
            'clay': 'clay',
            'sandy': 'sandy',
            'loam': 'loamy',
            'well-drained': 'well-drained',
            'fertile': 'fertile'
        }
        
        for word, soil_type in soil_keywords.items():
            if word in text.lower():
                return soil_type
        
        return "unknown"
    
    def save_model(self, path):
        """Save the trained model components."""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }, path)
    
    def load_model(self, path):
        """Load trained model components."""
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']