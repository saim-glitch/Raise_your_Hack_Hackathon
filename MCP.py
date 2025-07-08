# E-Commerce Application - Model Context Protocol Integration with Search and Dashboard

class Product:
    def __init__(self, id, name, price):
        self.id = id
        self.name = name
        self.price = price

    def __repr__(self):
        return f"Product(id={self.id}, name={self.name}, price=${self.price})"


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_product(self, product):
        self.items.append(product)

    def remove_product(self, product_id):
        self.items = [item for item in self.items if item.id != product_id]

    def calculate_total(self):
        return sum(product.price for product in self.items)

    def __repr__(self):
        return f"ShoppingCart(items={self.items})"


class Order:
    def __init__(self, cart):
        self.cart = cart
        self.total = self.cart.calculate_total()

    def process_order(self):
        return f"Order processed with total amount: ${self.total}"


class Store:
    def __init__(self):
        self.products = []
        self.cart = ShoppingCart()

    def add_product(self, product):
        self.products.append(product)

    def search_products(self, search_term):
        return [product for product in self.products if search_term.lower() in product.name.lower()]

    def get_dashboard(self):
        total_products = len(self.products)
        total_cart_value = self.cart.calculate_total()
        return {
            "total_products": total_products,
            "total_cart_value": total_cart_value,
            "cart_items": self.cart.items
        }


# Example Usage
if __name__ == "__main__":
    # Creating a store and adding products
    store = Store()
    store.add_product(Product(1, "meat", 1200))
    store.add_product(Product(2, "chicken", 800))
    store.add_product(Product(3, "beef", 400))
    store.add_product(Product(4,"fish" ,300))
    store.add_product(Product(5,"vegitables" ,200))
    store.add_product(Product(6,"fruits" ,300))
    store.add_product(Product(7,"dairy" ,300))
    store.add_product(Product(8,"bread" ,300))
    store.add_product(Product(9,"rice" ,300))
    store.add_product(Product(10,"oil" ,300))
    store.add_product(Product(11,"spices" ,300))

    # Searching for products
    search_result = store.search_products("meat")
    print("Search Results:", search_result)  # Should return the meat product

    # Adding products to the shopping cart
    store.cart.add_product(search_result[0])  # Add meat to cart

    print(store.cart)  # Output current state of the cart

    # Creating an order from the cart
    order = Order(store.cart)
    print(order.process_order())  # Process the order and print total 

    # Dashboard information
    dashboard_info = store.get_dashboard()
    print("Dashboard Info:", dashboard_info)  # Output total products and cart value
