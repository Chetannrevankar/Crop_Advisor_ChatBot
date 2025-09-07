"""
Enhanced Response generation module for creating appropriate answers to user queries.
Integrates with comprehensive database and considers regional/soil factors.
"""

import pandas as pd
import random
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
from config import DISCLAIMER, DEFAULT_RESPONSE

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates appropriate responses based on user query and extracted entities."""
    
    def __init__(self, crops_db: pd.DataFrame, regional_data: pd.DataFrame, soil_data: pd.DataFrame):
        """
        Initialize with data sources.
        
        Parameters:
            crops_db (DataFrame): Crop disease database
            regional_data (DataFrame): Regional compatibility data
            soil_data (DataFrame): Soil compatibility data
        """
        self.crops_db = crops_db
        self.regional_data = regional_data
        self.soil_data = soil_data
        
        # Clean and preprocess data
        self._clean_data()
        
    def _clean_data(self):
        """Clean and standardize data for better matching."""
        # Convert all text columns to lowercase for consistent matching
        text_columns = ['crop_name', 'symptom', 'why_occur', 'prevention_methods', 
                       'common_regions', 'soil_type']
        
        for col in text_columns:
            if col in self.crops_db.columns:
                self.crops_db[col] = self.crops_db[col].astype(str).str.lower()
        
        if 'crop_name' in self.regional_data.columns:
            self.regional_data['crop_name'] = self.regional_data['crop_name'].astype(str).str.lower()
        if 'region' in self.regional_data.columns:
            self.regional_data['region'] = self.regional_data['region'].astype(str).str.lower()
        
        if 'crop_name' in self.soil_data.columns:
            self.soil_data['crop_name'] = self.soil_data['crop_name'].astype(str).str.lower()
        if 'soil_type' in self.soil_data.columns:
            self.soil_data['soil_type'] = self.soil_data['soil_type'].astype(str).str.lower()
    
    def generate_response(self, entities: Dict[str, Any], similar_symptom_indices: Optional[List[int]] = None) -> str:
        """
        Generate response based on extracted entities and similar symptoms.
        
        Parameters:
            entities (dict): Extracted entities from user query
            similar_symptom_indices (list): Indices of similar symptoms from FAISS
            
        Returns:
            str: Generated response
        """
        try:
            intent = entities.get('intent', 'unknown')
            
            response_methods = {
                'disease_diagnosis': self._generate_diagnosis_response,
                'cultivation_advice': self._generate_cultivation_response,
                'region_compatibility': self._generate_region_response
            }
            
            response_method = response_methods.get(intent, self._handle_unknown_intent)
            
            # Pass only the required arguments based on method signature
            if intent == 'disease_diagnosis':
                return response_method(entities, similar_symptom_indices)
            else:
                return response_method(entities)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "❌ Sorry, I encountered an error processing your request. Please try again."
    
    def _generate_diagnosis_response(self, entities: Dict[str, Any], similar_symptom_indices: Optional[List[int]] = None) -> str:
        """Generate enhanced response for disease diagnosis queries."""
        # Safely extract and handle None values - FIXED HERE
        crop = entities.get('crop', '')
        symptoms = entities.get('symptoms', [])
        
        # Handle None values and convert to lowercase safely
        crop = crop.lower() if crop else ''
        symptoms = [s.lower() for s in symptoms if s] if symptoms else []
        
        if not crop:
            return "🌱 Please specify which crop you're asking about (e.g., 'tomato', 'rice', 'mango')."
        
        if not symptoms:
            return f"🌱 Please describe the symptoms you're observing on your {crop} plants (e.g., 'yellow leaves', 'brown spots')."
        
        # Filter database for the specified crop
        crop_data = self.crops_db[self.crops_db['crop_name'] == crop]
        
        if crop_data.empty:
            # Try to find similar crop names
            similar_crops = self._find_similar_crops(crop)
            if similar_crops:
                return f"❌ I don't have information about '{crop}'. Did you mean: {', '.join(similar_crops[:3])}?"
            return f"❌ I don't have information about {crop}. Try: tomato, rice, wheat, cotton, sugarcane, mango, or vegetables."
        
        # Enhanced matching logic
        matched_symptoms = []
        
        # First try exact symptom matching
        for symptom in symptoms:
            exact_matches = crop_data[
                crop_data['symptom'].str.contains(symptom, case=False, na=False)
            ]
            if not exact_matches.empty:
                matched_symptoms.extend(exact_matches.to_dict('records'))
        
        # If no exact matches, use similar symptoms from FAISS
        if not matched_symptoms and similar_symptom_indices:
            for idx in similar_symptom_indices:
                if idx < len(crop_data):
                    matched_symptoms.append(crop_data.iloc[idx].to_dict())
        
        # If still no matches, try partial matching
        if not matched_symptoms:
            for symptom in symptoms:
                partial_matches = crop_data[
                    crop_data['symptom'].str.contains('|'.join(symptom.split()), case=False, na=False)
                ]
                if not partial_matches.empty:
                    matched_symptoms.extend(partial_matches.to_dict('records'))
        
        # If still no matches, return cultivation advice
        if not matched_symptoms:
            return self._generate_cultivation_response(entities)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matched_symptoms:
            match_id = (match.get('crop_name', ''), match.get('symptom', ''))
            if match_id not in seen:
                seen.add(match_id)
                unique_matches.append(match)
        
        # Generate enhanced response
        response = f"🌱 CROP DIAGNOSIS REPORT\n\n"
        response += f"📋 Crop: {crop.title()}\n"
        response += f"🔍 Symptoms: {', '.join([s.title() for s in symptoms])}\n"
        
        region = entities.get('region')
        if region:
            response += f"📍 Region: {region.title()}\n"
        
        soil_type = entities.get('soil_type')
        if soil_type:
            response += f"🌱 Soil: {soil_type.title()}\n"
        
        response += "\n" + "="*50 + "\n\n"
        
        for i, match in enumerate(unique_matches[:3]):  # Limit to top 3 matches
            confidence = min(95, 90 - (i * 10))
            
            response += f"✅ Match {i+1} ({confidence}% confidence):\n"
            response += f"   Symptom: {match.get('symptom', 'N/A').title()}\n"
            response += f"   Cause: {match.get('why_occur', 'N/A')}\n"
            
            # Format prevention methods better
            prevention_methods = str(match.get('prevention_methods', '')).split(',')
            response += "   Treatment:\n"
            for j, method in enumerate([m.strip() for m in prevention_methods if m.strip()][:3], 1):
                response += f"     {j}. {method}\n"
            
            # Add common regions for this specific issue
            common_regions = str(match.get('common_regions', '')).split(';')
            if common_regions and common_regions[0].strip():
                response += f"   🌍 Common in: {', '.join([r.strip().title() for r in common_regions[:3]])}\n"
            
            response += "\n"
        
        # Add regional and soil considerations - FIXED None handling here too
        region = entities.get('region')
        soil_type = entities.get('soil_type')
        
        region_lower = region.lower() if region else ''
        soil_type_lower = soil_type.lower() if soil_type else ''
        
        if region_lower:
            regional_advice = self._get_regional_advice(crop, region_lower)
            if regional_advice:
                response += "📍 REGIONAL ADVICE:\n"
                response += f"   • {regional_advice}\n\n"
        
        if soil_type_lower:
            soil_advice = self._get_soil_advice(crop, soil_type_lower)
            if soil_advice:
                response += "🌱 SOIL RECOMMENDATIONS:\n"
                response += f"   • {soil_advice}\n\n"
        
        # Add seasonal advice if available
        if region_lower:
            seasonal_advice = self._get_seasonal_advice(crop, region_lower)
            if seasonal_advice:
                response += "📅 SEASONAL TIPS:\n"
                response += f"   • {seasonal_advice}\n\n"
        
        # Add general tips
        response += "💡 GENERAL TIPS:\n"
        response += "   • Monitor plants regularly for early detection\n"
        response += "   • Maintain proper plant spacing for air circulation\n"
        response += "   • Practice crop rotation to prevent disease buildup\n"
        response += "   • Test soil regularly for nutrient balance\n\n"
        
        response += "⚠️ DISCLAIMER:\n"
        response += "   This is AI-generated advice. For confirmed diagnosis and treatment,\n"
        response += "   consult local Krishi Vigyan Kendra or agricultural officers.\n"
        
        # Add reference ID
        ref_id = self._generate_reference_id(crop, symptoms, region)
        response += f"\n🔖 Reference ID: {ref_id}"
        
        return response
    
    def _generate_cultivation_response(self, entities: Dict[str, Any]) -> str:
        """Generate enhanced cultivation advice response."""
        # FIXED: Safe handling of None values
        crop = entities.get('crop', '')
        crop = crop.lower() if crop else ''
        
        if not crop:
            return "🌱 Please specify which crop you want cultivation advice for."
        
        crop_data = self.crops_db[self.crops_db['crop_name'] == crop]
        
        if crop_data.empty:
            similar_crops = self._find_similar_crops(crop)
            if similar_crops:
                return f"❌ I don't have cultivation information about '{crop}'. Did you mean: {', '.join(similar_crops[:3])}?"
            return f"❌ I don't have cultivation information about {crop}. Try: tomato, rice, wheat, cotton, sugarcane, mango, or grape."
        
        # Get unique prevention methods as general advice
        prevention_methods = set()
        for methods in crop_data['prevention_methods']:
            if pd.notna(methods):
                prevention_methods.update([m.strip() for m in str(methods).split(',') if m.strip()])
        
        response = f"🌱 CULTIVATION GUIDE: {crop.title()}\n\n"
        
        response += "📋 GROWING TIPS:\n"
        response += "   • Ensure proper spacing between plants\n"
        response += "   • Maintain consistent watering schedule\n"
        response += "   • Monitor regularly for pests and diseases\n"
        response += "   • Test soil before planting\n"
        response += "   • Use quality seeds/planting material\n\n"
        
        if prevention_methods:
            response += "🛡️ DISEASE PREVENTION:\n"
            for i, method in enumerate(list(prevention_methods)[:5], 1):
                response += f"   {i}. {method}\n"
        
        # Add regional and soil considerations - FIXED None handling
        region = entities.get('region', '')
        soil_type = entities.get('soil_type', '')
        
        region_lower = region.lower() if region else ''
        soil_type_lower = soil_type.lower() if soil_type else ''
        
        if region_lower:
            regional_advice = self._get_regional_advice(crop, region_lower)
            if regional_advice:
                response += f"\n📍 FOR {region.upper()}:\n"
                response += f"   • {regional_advice}\n"
        
        if soil_type_lower:
            soil_advice = self._get_soil_advice(crop, soil_type_lower)
            if soil_advice:
                response += f"\n🌱 SOIL TIPS:\n"
                response += f"   • {soil_advice}\n"
        
        # Add variety recommendations if available
        if region_lower:
            variety_advice = self._get_variety_recommendations(crop, region_lower)
            if variety_advice:
                response += f"\n🌿 RECOMMENDED VARIETIES:\n"
                response += f"   • {variety_advice}\n"
        
        response += "\n⚠️ DISCLAIMER:\n"
        response += "   Consult local agricultural experts for specific recommendations.\n"
        
        return response
    
    def _generate_region_response(self, entities: Dict[str, Any]) -> str:
        """Generate enhanced response for region compatibility queries."""
        # FIXED: Safe handling of None values
        region = entities.get('region', '')
        region = region.lower() if region else ''
        
        if not region or region == 'unknown':
            return "🌱 Please specify a region or state name (e.g., 'Karnataka', 'Bangalore', 'Mysore')."
        
        # Find crops suitable for the region
        try:
            regional_crops = self.regional_data[
                self.regional_data['region'].str.contains(region, case=False, na=False)
            ]
            
            if regional_crops.empty:
                similar_regions = self._find_similar_regions(region)
                if similar_regions:
                    return f"❌ I don't have crop compatibility information for '{region}'. Did you mean: {', '.join(similar_regions[:3])}?"
                return f"❌ I don't have crop compatibility information for {region}. Try: Karnataka, Bangalore, Mysore, or other Indian states."
            
            response = f"🌱 CROPS SUITABLE FOR {region.upper()}:\n\n"
            
            # Group by suitability
            suitability_groups = {
                'very high': regional_crops[regional_crops['suitability'].str.contains('very high', case=False, na=False)],
                'high': regional_crops[regional_crops['suitability'].str.contains('high', case=False, na=False)],
                'medium': regional_crops[regional_crops['suitability'].str.contains('medium', case=False, na=False)],
                'low': regional_crops[regional_crops['suitability'].str.contains('low', case=False, na=False)]
            }
            
            for suitability, group in suitability_groups.items():
                if not group.empty:
                    response += f"✅ {suitability.upper()} SUITABILITY:\n"
                    for _, row in group.iterrows():
                        crop_name = row.get('crop_name', 'N/A').title()
                        advice = row.get('advice', '')
                        response += f"   • {crop_name}: {advice}\n"
                    response += "\n"
            
            response += "💡 TIPS:\n"
            response += "   • Always test soil before planting\n"
            response += "   • Consider local climate variations\n"
            response += "   • Consult local agricultural officers\n\n"
            
            response += "⚠️ DISCLAIMER:\n"
            response += "   Local conditions may vary. Consult experts for specific advice.\n"
            
            return response
        except Exception as e:
            logger.error(f"Error processing region query: {e}")
            return f"❌ Error processing region query. Please try again with a different region name."
    
    def _handle_unknown_intent(self, entities: Dict[str, Any]) -> str:
        """Handle queries where intent cannot be determined."""
        crop = entities.get('crop')
        symptoms = entities.get('symptoms', [])
        
        if crop and not symptoms:
            return f"🌱 Please describe the symptoms you're observing on your {crop} plants."
        elif symptoms and not crop:
            return "🌱 Please specify which crop is showing these symptoms."
        else:
            return DEFAULT_RESPONSE
    
    def _get_regional_advice(self, crop: str, region: str) -> Optional[str]:
        """Get regional-specific advice for a crop."""
        try:
            regional_info = self.regional_data[
                (self.regional_data['crop_name'] == crop) &
                (self.regional_data['region'].str.contains(region, case=False, na=False))
            ]
            
            if not regional_info.empty:
                return regional_info.iloc[0].get('advice')
            return None
        except Exception as e:
            logger.error(f"Error getting regional advice: {e}")
            return None
    
    def _get_soil_advice(self, crop: str, soil_type: str) -> Optional[str]:
        """Get soil-specific advice for a crop."""
        try:
            soil_info = self.soil_data[
                (self.soil_data['crop_name'] == crop) &
                (self.soil_data['soil_type'].str.contains(soil_type, case=False, na=False))
            ]
            
            if not soil_info.empty:
                return soil_info.iloc[0].get('advice')
            return None
        except Exception as e:
            logger.error(f"Error getting soil advice: {e}")
            return None
    
    def _find_similar_crops(self, query_crop: str) -> List[str]:
        """Find similar crop names using fuzzy matching."""
        try:
            all_crops = self.crops_db['crop_name'].unique()
            similar_crops = []
            
            # Handle common variations
            crop_variations = {
                'grape': 'grapes',
                'tomato': 'tomatoes', 
                'potato': 'potatoes',
                'mango': 'mangoes',
                'chili': 'chilli',
                # Add more variations
            }
            
            # Check variation first
            variation = crop_variations.get(query_crop)
            if variation and variation in all_crops:
                return [variation.title()]
            
            # Then check contains matching
            for crop in all_crops:
                if (query_crop in crop or crop in query_crop or
                    query_crop == crop[:-1] or query_crop + 's' == crop or
                    query_crop + 'es' == crop):
                    similar_crops.append(crop.title())
            
            return list(set(similar_crops))[:3]  # Return top 3 unique matches
            
        except Exception as e:
            logger.error(f"Error finding similar crops: {e}")
            return []
    
    def _find_similar_regions(self, query_region: str) -> List[str]:
        """Find similar region names."""
        try:
            all_regions = self.regional_data['region'].unique()
            similar_regions = []
            
            for region in all_regions:
                if query_region in region or region in query_region:
                    similar_regions.append(region.title())
            
            return list(set(similar_regions))[:3]  # Return top 3 matches
            
        except Exception as e:
            logger.error(f"Error finding similar regions: {e}")
            return []
    
    def _get_seasonal_advice(self, crop: str, region: str) -> Optional[str]:
        """Get seasonal planting advice based on region and crop."""
        seasonal_data = {
            "tomato": {
                "karnataka": "Ideal planting season: June-July (Kharif) or October-November (Rabi)",
                "maharashtra": "Best planted in June-July for Kharif season",
                "default": "Generally planted in Kharif (June-July) or Rabi (October-November) seasons"
            },
            "rice": {
                "karnataka": "Transplant during monsoon season (June-July)",
                "andhra pradesh": "Kharif: June-July, Rabi: November-December",
                "default": "Main season: Kharif (June-July), Rabi in some regions (Nov-Dec)"
            },
            "cotton": {
                "karnataka": "Sow in May-June for Kharif season",
                "maharashtra": "Plant in June-July with onset of monsoon",
                "default": "Typically sown in Kharif season (May-July)"
            },
            "default": "Consult local agricultural calendar for optimal planting times"
        }
        
        return seasonal_data.get(crop, {}).get(region, seasonal_data.get(crop, {}).get('default', seasonal_data['default']))
    
    def _get_variety_recommendations(self, crop: str, region: str) -> Optional[str]:
        """Get variety recommendations based on region."""
        variety_data = {
            "tomato": {
                "karnataka": "Arka Vikas, Arka Abha, Pusa Ruby, Arka Rakshak",
                "maharashtra": "Arka Rakshak, NS-501, Pusa Ruby, Hybrid varieties",
                "default": "Local hybrid varieties adapted to regional conditions"
            },
            "rice": {
                "karnataka": "Jaya, IR-64, MTU-1010, Rajamudi, BPT-5204",
                "andhra pradesh": "BPT-5204, MTU-1001, Swarna, Samba Mahsuri",
                "default": "High-yielding varieties suitable for local conditions"
            },
            "cotton": {
                "karnataka": "Bunny, RCH-2, DCH-32",
                "maharashtra": "NH-44, Bunny, RCH-2",
                "default": "BT cotton varieties recommended for most regions"
            },
            "default": "Consult local agriculture department for region-specific variety recommendations"
        }
        
        return variety_data.get(crop, {}).get(region, variety_data.get(crop, {}).get('default', variety_data['default']))
    
    def _generate_reference_id(self, crop: str, symptoms: List[str], region: Optional[str] = None) -> str:
        """Generate a unique reference ID."""
        crop_code = crop.upper()[:4] if crop else "CROP"
        symptom_code = symptoms[0].upper()[:4] if symptoms and len(symptoms) > 0 else "GEN"
        region_code = region.upper()[:2] if region and region != "unknown" else "XX"
        date_code = datetime.now().strftime("%Y%m%d")
        random_code = str(random.randint(100, 999))
        return f"{crop_code}-{symptom_code}-{region_code}-{date_code}-{random_code}"