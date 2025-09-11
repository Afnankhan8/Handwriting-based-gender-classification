import mysql.connector
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        return connection
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        return None

def create_database():
    """Create database if it doesn't exist"""
    try:
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME}")
            logger.info(f"Database '{Config.DB_NAME}' created or already exists")
            
            cursor.close()
            connection.close()
            return True
    except mysql.connector.Error as err:
        logger.error(f"Error creating database: {err}")
        return False

def create_tables():
    """Create necessary tables"""
    try:
        # First create database
        if not create_database():
            return False
        
        # Connect to the specific database
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
        
        cursor = connection.cursor()
        
        # Create predictions table
        predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            predicted_gender ENUM('Male', 'Female') NOT NULL,
            confidence DECIMAL(5,2) NOT NULL,
            algorithm VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            features_count INT DEFAULT 0,
            processing_time DECIMAL(6,3) DEFAULT 0.0,
            INDEX idx_timestamp (timestamp),
            INDEX idx_gender (predicted_gender),
            INDEX idx_algorithm (algorithm)
        )
        """
        
        cursor.execute(predictions_table)
        logger.info("Predictions table created or already exists")
        
        # Create users table (for future authentication)
        users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('user', 'admin') DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            is_active BOOLEAN DEFAULT TRUE,
            INDEX idx_username (username),
            INDEX idx_email (email)
        )
        """
        
        cursor.execute(users_table)
        logger.info("Users table created or already exists")
        
        # Create handwriting_samples table (for training data)
        samples_table = """
        CREATE TABLE IF NOT EXISTS handwriting_samples (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            actual_gender ENUM('Male', 'Female') NOT NULL,
            script_type VARCHAR(50) DEFAULT 'Gurumukhi',
            writer_age INT DEFAULT NULL,
            writer_id VARCHAR(50) DEFAULT NULL,
            image_path VARCHAR(500) NOT NULL,
            features_extracted JSON DEFAULT NULL,
            is_training_data BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_gender (actual_gender),
            INDEX idx_writer (writer_id),
            INDEX idx_script (script_type)
        )
        """
        
        cursor.execute(samples_table)
        logger.info("Handwriting samples table created or already exists")
        
        # Create model_performance table
        performance_table = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            accuracy DECIMAL(5,2) NOT NULL,
            precision_score DECIMAL(5,2) DEFAULT NULL,
            recall_score DECIMAL(5,2) DEFAULT NULL,
            f1_score DECIMAL(5,2) DEFAULT NULL,
            training_samples INT DEFAULT 0,
            test_samples INT DEFAULT 0,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hyperparameters JSON DEFAULT NULL,
            INDEX idx_model (model_name),
            INDEX idx_accuracy (accuracy)
        )
        """
        
        cursor.execute(performance_table)
        logger.info("Model performance table created or already exists")
        
        # Create feature_extraction_logs table
        features_table = """
        CREATE TABLE IF NOT EXISTS feature_extraction_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_filename VARCHAR(255) NOT NULL,
            extraction_method VARCHAR(100) NOT NULL,
            features_count INT NOT NULL,
            extraction_time DECIMAL(6,3) NOT NULL,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_method (extraction_method),
            INDEX idx_success (success)
        )
        """
        
        cursor.execute(features_table)
        logger.info("Feature extraction logs table created or already exists")
        
        # Insert sample data for demonstration
        insert_sample_data(cursor)
        
        # Commit changes
        connection.commit()
        logger.info("All tables created successfully")
        
        cursor.close()
        connection.close()
        return True
        
    except mysql.connector.Error as err:
        logger.error(f"Error creating tables: {err}")
        return False

def insert_sample_data(cursor):
    """Insert sample data for demonstration"""
    try:
        # Check if model performance data exists
        cursor.execute("SELECT COUNT(*) FROM model_performance")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert sample model performance data
            sample_models = [
                ('RandomForest', 94.6, 94.2, 95.1, 94.6, 160, 40),
                ('KNN', 89.3, 88.7, 90.2, 89.4, 160, 40),
                ('DecisionTree', 87.5, 86.9, 88.3, 87.6, 160, 40),
                ('AdaBoost', 91.2, 90.8, 91.7, 91.2, 160, 40)
            ]
            
            insert_query = """
            INSERT INTO model_performance 
            (model_name, accuracy, precision_score, recall_score, f1_score, training_samples, test_samples)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.executemany(insert_query, sample_models)
            logger.info("Sample model performance data inserted")
        
        # Insert sample admin user (password: admin123)
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            from werkzeug.security import generate_password_hash
            admin_password = generate_password_hash('admin123')
            
            admin_user = (
                'admin', 
                'admin@handwriteai.com', 
                admin_password, 
                'admin'
            )
            
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
            """, admin_user)
            
            logger.info("Admin user created (username: admin, password: admin123)")
        
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")

def reset_database():
    """Reset database (drop and recreate all tables)"""
    try:
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
        
        cursor = connection.cursor()
        
        # Drop tables in reverse order (due to foreign key constraints)
        tables = [
            'feature_extraction_logs',
            'model_performance', 
            'handwriting_samples',
            'predictions',
            'users'
        ]
        
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Table '{table}' dropped")
        
        cursor.close()
        connection.close()
        
        # Recreate tables
        return create_tables()
        
    except mysql.connector.Error as err:
        logger.error(f"Error resetting database: {err}")
        return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'reset':
        logger.info("Resetting database...")
        if reset_database():
            logger.info("Database reset completed successfully")
        else:
            logger.error("Database reset failed")
    else:
        logger.info("Creating database tables...")
        if create_tables():
            logger.info("Database setup completed successfully")
        else:
            logger.error("Database setup failed")