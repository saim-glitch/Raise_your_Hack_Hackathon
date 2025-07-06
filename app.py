from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random
import re
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from typing import Dict, List, Optional

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
XAI_API_KEY = "xai-BVC5I7NtvhNj9gnzVlab2tj7jQ4SRvq0tznruHfPL70BRbUWEh2VwoHJq0M7mXu58nIpDE7qhZZ8ZKAv"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'your-email@gmail.com',  # Replace with your email
    'password': 'your-app-password',   # Replace with your app password
    'sender_name': 'Metro Store AI Assistant'
}

# AI Assistant Configuration
AI_ASSISTANT_CONFIG = {
    'name': 'Grok Assistant',
    'personality': 'friendly, helpful, and conversational like a real human',
    'capabilities': [
        'Natural conversation with customers',
        'Product recommendations from store inventory',
        'Place orders directly through chat',
        'Send email confirmations',
        'Contextual product suggestions',
        'Availability checking',
        'Price comparisons'
    ]
}

class EmailService:
    """Handle email notifications"""
    
    def __init__(self, config):
        self.config = config
    
    def send_order_confirmation(self, user_email, user_name, order_details):
        """Send order confirmation email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.config['sender_name']} <{self.config['email']}>"
            msg['To'] = user_email
            msg['Subject'] = "Order Confirmation - Metro Store"
            
            # Create HTML email content
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: #4CAF50; color: white; padding: 20px; text-align: center;">
                    <h1>Order Confirmed! 🎉</h1>
                </div>
                
                <div style="padding: 20px;">
                    <h2>Hello {user_name}!</h2>
                    <p>Thank you for your order. Your AI Assistant has successfully processed your request.</p>
                    
                    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3>Order Details:</h3>
                        <p><strong>Order ID:</strong> {order_details['order_id']}</p>
                        <p><strong>Order Date:</strong> {order_details['order_date']}</p>
                        <p><strong>Total Amount:</strong> Rs. {order_details['total_amount']}</p>
                    </div>
                    
                    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px;">
                        <h3>Items Ordered:</h3>
                        <ul>
            """
            
            for item in order_details['items']:
                html_content += f"""
                    <li>{item['name']} by {item['brand']} - Rs. {item['price']}</li>
                """
            
            html_content += """
                        </ul>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 20px; background-color: #e8f5e8; border-radius: 5px;">
                        <h3>What's Next?</h3>
                        <p>• Your order is being prepared</p>
                        <p>• You'll receive updates via email</p>
                        <p>• Expected delivery: 2-3 business days</p>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <p>Thank you for shopping with Metro Store!</p>
                        <p><em>Your AI Assistant is always here to help 🤖</em></p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email'], self.config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Email sending failed: {str(e)}")
            return False

class GrokAIAssistant:
    """Enhanced AI Assistant with Grok API integration"""
    
    def __init__(self, df, email_service):
        self.df = df
        self.email_service = email_service
        self.conversation_history = []
        self.setup_product_context()
        
    def setup_product_context(self):
        """Prepare product data for AI context"""
        # Create a searchable product database
        self.df['search_text'] = (
            self.df['name'].fillna('') + ' ' + 
            self.df['brand'].fillna('') + ' ' + 
            self.df['description'].fillna('') + ' ' + 
            self.df['category'].fillna('')
        ).str.lower()
        
        # Create product summary for AI context
        self.product_summary = self.create_product_summary()
        
    def create_product_summary(self):
        """Create a concise product summary for AI context"""
        categories = self.df['category'].value_counts().head(10).to_dict()
        brands = self.df['brand'].value_counts().head(10).to_dict()
        
        summary = {
            'total_products': len(self.df),
            'categories': categories,
            'brands': brands,
            'price_range': {
                'min': self.df['price_numeric'].min(),
                'max': self.df['price_numeric'].max(),
                'average': self.df['price_numeric'].mean()
            }
        }
        
        return summary
    
    def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """Search products based on query"""
        query_lower = query.lower()
        
        # Search in product text
        matching_products = self.df[
            self.df['search_text'].str.contains(query_lower, case=False, na=False)
        ]
        
        if len(matching_products) == 0:
            # Try broader search
            words = query_lower.split()
            for word in words:
                if len(word) > 2:  # Only search meaningful words
                    matches = self.df[
                        self.df['search_text'].str.contains(word, case=False, na=False)
                    ]
                    if len(matches) > 0:
                        matching_products = matches
                        break
        
        # Convert to list of dictionaries
        products = []
        for _, product in matching_products.head(limit).iterrows():
            products.append({
                'id': str(product.name),
                'name': product['name'],
                'brand': product['brand'],
                'price': product['price_numeric'],
                'category': product['category'],
                'description': product.get('description', ''),
                'availability': product.get('availability', 'Check availability'),
                'image_url': product.get('image_url', 'https://via.placeholder.com/300')
            })
        
        return products
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product details by ID"""
        try:
            product = self.df.iloc[int(product_id)]
            return {
                'id': product_id,
                'name': product['name'],
                'brand': product['brand'],
                'price': product['price_numeric'],
                'category': product['category'],
                'description': product.get('description', ''),
                'availability': product.get('availability', 'Check availability'),
                'image_url': product.get('image_url', 'https://via.placeholder.com/300')
            }
        except:
            return None
    
    def create_system_prompt(self) -> str:
        """Create system prompt for Grok API"""
        return f"""You are a friendly AI shopping assistant for Metro Store. You talk like a real human - warm, helpful, and conversational.

STORE INVENTORY OVERVIEW:
- Total Products: {self.product_summary['total_products']}
- Categories: {', '.join(self.product_summary['categories'].keys())}
- Top Brands: {', '.join(list(self.product_summary['brands'].keys())[:5])}
- Price Range: Rs. {self.product_summary['price_range']['min']:.0f} - Rs. {self.product_summary['price_range']['max']:.0f}

CAPABILITIES:
1. Search and recommend products from our store inventory
2. Place orders directly when customers ask
3. Send email confirmations for orders
4. Provide product information, availability, and pricing
5. Have natural conversations about shopping needs

IMPORTANT GUIDELINES:
- Always search the store inventory when customers ask about products
- When customers want to order/buy something, confirm details and place the order
- Be conversational and friendly, like talking to a friend
- Provide helpful suggestions based on customer needs
- Always mention product availability and pricing
- Use emojis naturally in conversation
- When placing orders, confirm: product name, quantity, and total price

ORDERING PROCESS:
When a customer wants to order something:
1. Confirm the product details
2. Ask for quantity if not specified
3. Calculate total price
4. Confirm the order
5. Mention that email confirmation will be sent

Remember: You can access real product data and place actual orders. Be helpful and human-like!"""

    def call_grok_api(self, user_message: str, user_profile: Dict) -> Dict:
        """Call Grok API for natural conversation"""
        try:
            # First, search for products based on user message
            search_keywords = self.extract_search_keywords(user_message)
            products = []
            
            if search_keywords:
                products = self.search_products(search_keywords, limit=5)
            
            # Prepare conversation context
            messages = [
                {"role": "system", "content": self.create_system_prompt()}
            ]
            
            # Add product context if found
            if products:
                product_context = "AVAILABLE PRODUCTS FOR THIS QUERY:\n"
                for i, product in enumerate(products, 1):
                    product_context += f"{i}. {product['name']} by {product['brand']} - Rs. {product['price']} ({product['category']})\n"
                messages.append({"role": "system", "content": product_context})
            
            # Add recent conversation history
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": entry['user_message']})
                messages.append({"role": "assistant", "content": entry['ai_response']})
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grok-beta",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # Make API call
            response = requests.post(XAI_API_URL, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Process the response for actions
                return self.process_ai_response(ai_response, user_message, user_profile, products)
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                return self.fallback_response(user_message, user_profile)
                
        except Exception as e:
            print(f"Grok API error: {str(e)}")
            return self.fallback_response(user_message, user_profile)
    
    def process_ai_response(self, ai_response: str, user_message: str, user_profile: Dict, products: List[Dict]) -> Dict:
        """Process AI response and extract actions"""
        # Check if user wants to place an order
        order_intent = self.detect_order_intent(user_message)
        order_result = None
        
        if order_intent and products:
            order_result = self.process_order(products[0], user_profile, user_message)
        
        return {
            'response': ai_response,
            'products': products,
            'order_placed': order_result is not None,
            'order_details': order_result,
            'context': 'grok_api'
        }
    
    def extract_search_keywords(self, message: str) -> str:
        """Extract product search keywords from message"""
        # Simple keyword extraction - enhanced logic
        search_indicators = [
            'looking for', 'want to buy', 'need', 'search for',
            'show me', 'find', 'get me', 'order', 'buy', 'want'
        ]
        
        message_lower = message.lower()
        
        # Check for direct product mentions
        product_keywords = ['meat', 'chicken', 'beef', 'fish', 'vegetables', 'fruits', 'dairy', 'bread', 'rice', 'oil', 'spices']
        for keyword in product_keywords:
            if keyword in message_lower:
                return keyword
        
        # Check for search indicators
        for indicator in search_indicators:
            if indicator in message_lower:
                # Extract the part after the indicator
                parts = message_lower.split(indicator, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # If no specific indicator, return the whole message if it seems like a search
        if len(message.split()) <= 3:  # Short messages are likely search queries
            return message
        
        return ""
    
    def detect_order_intent(self, message: str) -> bool:
        """Detect if user wants to place an order"""
        order_keywords = [
            'order', 'buy', 'purchase', 'get me', 'i want to buy',
            'add to cart', 'place order', 'i need to order'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in order_keywords)
    
    def process_order(self, product: Dict, user_profile: Dict, user_message: str) -> Dict:
        """Process order placement"""
        try:
            # Extract quantity from message (default to 1)
            quantity = self.extract_quantity(user_message)
            
            # Calculate total
            total_amount = product['price'] * quantity
            
            # Create order details
            order_details = {
                'order_id': f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'order_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'customer_name': user_profile.get('name', 'Customer'),
                'customer_email': user_profile.get('email', ''),
                'items': [{
                    'name': product['name'],
                    'brand': product['brand'],
                    'price': product['price'],
                    'quantity': quantity
                }],
                'total_amount': total_amount,
                'status': 'confirmed'
            }
            
            # Send email confirmation if email is available
            if user_profile.get('email'):
                email_sent = self.email_service.send_order_confirmation(
                    user_profile['email'],
                    user_profile.get('name', 'Customer'),
                    order_details
                )
                order_details['email_sent'] = email_sent
            
            return order_details
            
        except Exception as e:
            print(f"Order processing error: {str(e)}")
            return None
    
    def extract_quantity(self, message: str) -> int:
        """Extract quantity from message"""
        # Look for numbers in the message
        numbers = re.findall(r'\b(\d+)\b', message)
        if numbers:
            return int(numbers[0])
        return 1  # Default quantity
    
    def fallback_response(self, user_message: str, user_profile: Dict) -> Dict:
        """Enhanced fallback response when API fails"""
        # Use simple keyword matching as fallback
        search_keywords = self.extract_search_keywords(user_message)
        products = []
        
        if search_keywords:
            products = self.search_products(search_keywords, limit=3)
        
        # Generate appropriate response based on user message
        message_lower = user_message.lower()
        
        if 'hello' in message_lower or 'hi' in message_lower:
            response = "Hello! 👋 Welcome to Metro Store! I'm your AI shopping assistant. I can help you find products, check prices, and even place orders. What are you looking for today?"
        elif 'meat' in message_lower:
            if products:
                response = "Great! I found some fresh meat products for you:\n\n"
                for i, product in enumerate(products, 1):
                    response += f"{i}. **{product['name']}** by {product['brand']} - Rs. {product['price']} ({product['availability']})\n"
                response += "\nWould you like to order any of these or need more details? 🥩"
            else:
                response = "I'd love to help you find meat products! Let me search our fresh meat section for you. We have chicken, beef, mutton, and fish available. What specific type of meat are you looking for?"
        elif any(word in message_lower for word in ['yes', 'show', 'details']):
            if products:
                response = "Here are the product details:\n\n"
                for i, product in enumerate(products, 1):
                    response += f"**{i}. {product['name']}**\n"
                    response += f"   Brand: {product['brand']}\n"
                    response += f"   Price: Rs. {product['price']}\n"
                    response += f"   Category: {product['category']}\n"
                    response += f"   Status: {product['availability']}\n\n"
                response += "Would you like to add any of these to your cart or place an order? 🛒"
            else:
                response = "I'd be happy to show you details! Could you tell me what specific products you're interested in? I can search our entire inventory for you."
        else:
            if products:
                response = f"I found some products related to '{search_keywords}':\n\n"
                for i, product in enumerate(products, 1):
                    response += f"{i}. **{product['name']}** by {product['brand']} - Rs. {product['price']}\n"
                response += "\nWould you like more details or want to order any of these? 😊"
            else:
                response = "I'm here to help you find the perfect products! 🛍️ You can ask me to:\n\n"
                response += "• Search for specific products (e.g., 'show me chicken')\n"
                response += "• Check prices and availability\n"
                response += "• Place orders directly\n"
                response += "• Get product recommendations\n\n"
                response += "What would you like to find today?"
        
        return {
            'response': response,
            'products': products,
            'order_placed': False,
            'context': 'fallback'
        }

# Utility functions (keeping existing ones)
def parse_price(price_str):
    """Parse price string and convert to float"""
    if pd.isna(price_str):
        return 0.0
    
    price_str = str(price_str)
    price_str = price_str.replace('Rs.', '').replace('₹', '').replace(',', '').strip()
    
    try:
        numbers = re.findall(r'\d+\.?\d*', price_str)
        if numbers:
            return float(numbers[0])
        else:
            return 0.0
    except:
        return 0.0

def safe_str(value, max_length=None):
    """Safely convert value to string with optional length limit"""
    if pd.isna(value):
        return ""
    str_val = str(value)
    if max_length and len(str_val) > max_length:
        return str_val[:max_length]
    return str_val

# Load and process data
def load_data():
    """Load the CSV data"""
    try:
        df = pd.read_csv('metro-data.csv')
        
        # Clean and process the data
        df['price_numeric'] = df['price'].apply(parse_price)
        df['name'] = df['name'].fillna('Unknown Product')
        df['brand'] = df['brand'].fillna('Unknown Brand')
        df['category'] = df['category'].fillna('General')
        df['description'] = df['description'].fillna('No description available')
        df['availability'] = df['availability'].fillna('Check Availability')
        
        return df
    except FileNotFoundError:
        # Create sample data structure with more meat products
        sample_data = {
            'url': ['https://example1.com'] * 60,
            'name': [
                'Chicken Breast Fresh', 'Mutton Boneless', 'Fish Fillet Fresh', 'Beef Steak Cut',
                'Chicken Leg Piece', 'Prawns Fresh', 'Mutton Chops', 'Chicken Wings',
                'Whey Protein Powder', 'Energy Drink', 'Organic Rice', 'Cooking Oil Premium',
                'Fresh Vegetables', 'Dairy Milk', 'Bread Loaf', 'Basmati Rice'
            ] * 4,
            'price': [
                'Rs. 500', 'Rs. 800', 'Rs. 600', 'Rs. 900',
                'Rs. 450', 'Rs. 700', 'Rs. 750', 'Rs. 400',
                'Rs. 2000', 'Rs. 150', 'Rs. 1000', 'Rs. 800',
                'Rs. 200', 'Rs. 60', 'Rs. 40', 'Rs. 120'
            ] * 4,
            'brand': [
                'Fresh Mart', 'Premium Meats', 'Ocean Fresh', 'Butcher\'s Choice',
                'Farm Fresh', 'Sea Harvest', 'Premium Cuts', 'Fresh Mart',
                'Optimum', 'Red Bull', 'Organic India', 'Fortune',
                'Farm Fresh', 'Amul', 'Britannia', 'India Gate'
            ] * 4,
            'image_url': ['https://via.placeholder.com/300'] * 64,
            'availability': ['In Stock', 'Available', 'Limited Stock', 'Fresh Daily'] * 16,
            'description': [
                'Fresh chicken breast high in protein', 'Premium mutton boneless cuts', 'Fresh fish fillet daily catch', 'Premium beef steak cuts',
                'Fresh chicken leg pieces', 'Fresh prawns from coast', 'Premium mutton chops', 'Fresh chicken wings',
                'Premium whey protein for workouts', 'Energy boost drink', 'Organic basmati rice', 'Premium cooking oil',
                'Fresh daily vegetables', 'Pure dairy milk', 'Fresh bread loaf', 'Premium basmati rice'
            ] * 4,
            'category': [
                'Meat', 'Meat', 'Meat', 'Meat',
                'Meat', 'Seafood', 'Meat', 'Meat',
                'Supplements', 'Beverages', 'Grains', 'Cooking Oil',
                'Vegetables', 'Dairy', 'Bakery', 'Grains'
            ] * 4
        }
        df = pd.DataFrame(sample_data)
        # Calculate numeric prices
        df['price_numeric'] = df['price'].apply(parse_price)
        return df

# Initialize services
df = load_data()
email_service = EmailService(EMAIL_CONFIG)
ai_assistant = GrokAIAssistant(df, email_service)

# Initialize session data
def init_session():
    if 'user_profile' not in session:
        session['user_profile'] = {
            'logged_in': False,
            'name': '',
            'phone': '',
            'email': '',
            'age': 25,
            'gender': 'Male',
            'location': 'Mumbai',
            'preferences': []
        }
    
    if 'cart' not in session:
        session['cart'] = []
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if 'orders' not in session:
        session['orders'] = []

# Routes
@app.route('/')
def home():
    init_session()
    
    # Get featured products
    featured_products = df.sample(n=min(6, len(df)))
    
    # Convert to dict for template
    featured_products_list = []
    for _, product in featured_products.iterrows():
        featured_products_list.append({
            'name': safe_str(product['name']),
            'brand': safe_str(product['brand']),
            'price': product['price_numeric'],
            'image_url': product.get('image_url', 'https://via.placeholder.com/300'),
            'description': safe_str(product.get('description', ''), 100),
            'id': str(product.name)
        })
    
    # Statistics
    stats = {
        'total_products': len(df),
        'total_brands': df['brand'].nunique(),
        'total_categories': df['category'].nunique(),
        'cart_items': len(session['cart']),
        'total_orders': len(session['orders'])
    }
    
    return render_template('home.html', 
                         user=session['user_profile'],
                         featured_products=featured_products_list,
                         stats=stats,
                         ai_config=AI_ASSISTANT_CONFIG)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user_profile'] = {
            'logged_in': True,
            'name': request.form['name'],
            'phone': request.form['phone'],
            'email': request.form['email'],
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'location': request.form['location'],
            'preferences': []
        }
        flash(f"Welcome {request.form['name']}! Your AI Assistant is ready to help you shop! 🛍️", 'success')
        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session['user_profile']['logged_in'] = False
    flash('You have been logged out. Come back soon! 👋', 'info')
    return redirect(url_for('home'))

@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    """Enhanced AI Chat endpoint with Grok API"""
    init_session()
    
    query = request.json.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': 'Please enter a message'})
    
    try:
        # Get AI response from Grok API
        ai_response = ai_assistant.call_grok_api(query, session['user_profile'])
        
        # Handle order placement
        if ai_response.get('order_placed'):
            order_details = ai_response['order_details']
            session['orders'].append(order_details)
            session.modified = True
            
            # Add order confirmation to response
            ai_response['response'] += f"\n\n✅ **Order Confirmed!** 🎉\n"
            ai_response['response'] += f"📧 Email confirmation sent to {session['user_profile'].get('email', 'your email')}\n"
            ai_response['response'] += f"🆔 Order ID: {order_details['order_id']}"
        
        # Save to chat history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': query,
            'ai_response': ai_response['response'],
            'context': ai_response.get('context', 'grok_api'),
            'products_found': len(ai_response.get('products', [])),
            'order_placed': ai_response.get('order_placed', False)
        }
        
        ai_assistant.conversation_history.append(chat_entry)
        session['chat_history'].append(chat_entry)
        session.modified = True
        
        # Format response
        response_data = {
            'success': True,
            'message': ai_response['response'],
            'products': ai_response.get('products', []),
            'context': ai_response.get('context', 'grok_api'),
            'order_placed': ai_response.get('order_placed', False),
            'order_details': ai_response.get('order_details'),
            'timestamp': datetime.now().strftime('%H:%M')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Sorry, I encountered an error: {str(e)}. Please try again.',
            'timestamp': datetime.now().strftime('%H:%M')
        })

@app.route('/orders')
def orders():
    """View order history"""
    init_session()
    
    return render_template('orders.html',
                         user=session['user_profile'],
                         orders=session['orders'])

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    init_session()
    
    product_data = request.json
    
    if not product_data:
        return jsonify({'success': False, 'message': 'No product data provided'})
    
    try:
        # Add product to cart
        cart_item = {
            'id': product_data.get('id'),
            'name': product_data.get('name'),
            'brand': product_data.get('brand'),
            'price': product_data.get('price'),
            'quantity': product_data.get('quantity', 1),
            'image_url': product_data.get('image_url', 'https://via.placeholder.com/300'),
            'added_at': datetime.now().isoformat()
        }
        
        session['cart'].append(cart_item)
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f"✅ {product_data.get('name')} added to cart!",
            'cart_count': len(session['cart'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error adding to cart: {str(e)}'})

@app.route('/cart')
def cart():
    """View cart"""
    init_session()
    
    # Calculate total
    total = sum(item['price'] * item['quantity'] for item in session['cart'])
    
    return render_template('cart.html',
                         user=session['user_profile'],
                         cart_items=session['cart'],
                         total=total)

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    init_session()
    
    item_id = request.json.get('id')
    
    if not item_id:
        return jsonify({'success': False, 'message': 'No item ID provided'})
    
    try:
        # Remove item from cart
        session['cart'] = [item for item in session['cart'] if item['id'] != item_id]
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': 'Item removed from cart',
            'cart_count': len(session['cart'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error removing item: {str(e)}'})

@app.route('/products')
def products():
    """View all products"""
    init_session()
    
    # Get filters
    category = request.args.get('category', '')
    brand = request.args.get('brand', '')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    search = request.args.get('search', '')
    
    # Filter products
    filtered_df = df.copy()
    
    if category:
        filtered_df = filtered_df[filtered_df['category'].str.contains(category, case=False, na=False)]
    
    if brand:
        filtered_df = filtered_df[filtered_df['brand'].str.contains(brand, case=False, na=False)]
    
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['price_numeric'] >= min_price]
    
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['price_numeric'] <= max_price]
    
    if search:
        filtered_df = filtered_df[
            filtered_df['name'].str.contains(search, case=False, na=False) |
            filtered_df['brand'].str.contains(search, case=False, na=False) |
            filtered_df['description'].str.contains(search, case=False, na=False)
        ]
    
    # Convert to list
    products_list = []
    for _, product in filtered_df.iterrows():
        products_list.append({
            'id': str(product.name),
            'name': safe_str(product['name']),
            'brand': safe_str(product['brand']),
            'price': product['price_numeric'],
            'category': safe_str(product['category']),
            'description': safe_str(product.get('description', ''), 100),
            'image_url': product.get('image_url', 'https://via.placeholder.com/300'),
            'availability': safe_str(product.get('availability', 'Check Availability'))
        })
    
    # Get unique categories and brands for filters
    categories = sorted(df['category'].dropna().unique())
    brands = sorted(df['brand'].dropna().unique())
    
    return render_template('products.html',
                         user=session['user_profile'],
                         products=products_list,
                         categories=categories,
                         brands=brands,
                         filters={
                             'category': category,
                             'brand': brand,
                             'min_price': min_price,
                             'max_price': max_price,
                             'search': search
                         })

@app.route('/product/<product_id>')
def product_detail(product_id):
    """View product details"""
    init_session()
    
    try:
        product = df.iloc[int(product_id)]
        
        product_data = {
            'id': product_id,
            'name': safe_str(product['name']),
            'brand': safe_str(product['brand']),
            'price': product['price_numeric'],
            'category': safe_str(product['category']),
            'description': safe_str(product.get('description', '')),
            'image_url': product.get('image_url', 'https://via.placeholder.com/300'),
            'availability': safe_str(product.get('availability', 'Check Availability'))
        }
        
        # Get similar products
        similar_products = df[df['category'] == product['category']].sample(n=min(4, len(df)))
        similar_list = []
        for _, sim_product in similar_products.iterrows():
            if str(sim_product.name) != product_id:
                similar_list.append({
                    'id': str(sim_product.name),
                    'name': safe_str(sim_product['name']),
                    'brand': safe_str(sim_product['brand']),
                    'price': sim_product['price_numeric'],
                    'image_url': sim_product.get('image_url', 'https://via.placeholder.com/300')
                })
        
        return render_template('product_detail.html',
                             user=session['user_profile'],
                             product=product_data,
                             similar_products=similar_list)
        
    except (IndexError, ValueError):
        flash('Product not found', 'error')
        return redirect(url_for('products'))

@app.route('/analytics')
def analytics():
    """View analytics dashboard"""
    init_session()
    
    # Create analytics data
    analytics_data = {
        'total_products': len(df),
        'total_brands': df['brand'].nunique(),
        'total_categories': df['category'].nunique(),
        'avg_price': df['price_numeric'].mean(),
        'price_range': {
            'min': df['price_numeric'].min(),
            'max': df['price_numeric'].max()
        },
        'category_distribution': df['category'].value_counts().to_dict(),
        'brand_distribution': df['brand'].value_counts().head(10).to_dict(),
        'user_stats': {
            'total_orders': len(session['orders']),
            'cart_items': len(session['cart']),
            'chat_messages': len(session['chat_history'])
        }
    }
    
    return render_template('analytics.html',
                         user=session['user_profile'],
                         analytics=analytics_data)

@app.route('/api/products/search')
def api_search_products():
    """API endpoint for product search"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify({'success': False, 'message': 'No query provided'})
    
    try:
        products = ai_assistant.search_products(query, limit)
        return jsonify({
            'success': True,
            'products': products,
            'count': len(products)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify({
        'total_products': len(df),
        'total_brands': df['brand'].nunique(),
        'total_categories': df['category'].nunique(),
        'cart_items': len(session.get('cart', [])),
        'total_orders': len(session.get('orders', []))
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)