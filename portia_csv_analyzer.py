import os
import csv
import json
import time
import pandas as pd
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get configuration from environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
CSV_SAMPLE_SIZE = int(os.getenv("CSV_SAMPLE_SIZE", "5"))
CSV_INSIGHT_THRESHOLD = int(os.getenv("CSV_INSIGHT_THRESHOLD", "10"))

# Import Portia SDK with proper imports
try:
    from portia import (
        Portia,
        Plan,
        PlanRun,
        PlanRunState,
        default_config,
        StorageClass
    )
    
    # Initialize default Portia config
    portia_config = default_config()
    # Use memory storage for simplicity
    portia_config.storage_class = StorageClass.MEMORY
    
    # Flag if Portia is available
    PORTIA_AVAILABLE = True
    print("Portia SDK initialized successfully")
except ImportError as e:
    print(f"Error importing Portia SDK: {e}")
    print("Please install Portia SDK: pip install portia-sdk")
    PORTIA_AVAILABLE = False

class PortiaCsvAnalyzer:
    """
    A CSV analyzer that uses Portia's SDK for planning and execution
    while keeping the raw data local for privacy
    """
    
    def __init__(self, csv_data):
        """
        Initialize the analyzer with CSV data
        
        Args:
            csv_data (str): CSV data as string
        """
        self.csv_data = csv_data
        self.thinking_log = []  # Initialize thinking log
        self.parsed_data = self._parse_csv(csv_data)
        self.analysis_results = None
        
        # Initialize Portia client if available
        if PORTIA_AVAILABLE:
            try:
                # Create Portia client
                self.portia = Portia(config=portia_config)
                print("Portia client created successfully")
            except Exception as e:
                print(f"Error creating Portia client: {e}")
                self.portia = None
        else:
            self.portia = None
    
    def _log_thinking(self, message, type="info"):
        """
        Log the thinking process for visualization
        
        Args:
            message (str): The thinking process message
            type (str): Type of thinking (info, plan, execute, insight)
        """
        # Make sure thinking_log is initialized
        if not hasattr(self, 'thinking_log'):
            self.thinking_log = []
            
        timestamp = time.strftime("%H:%M:%S")
        thinking_entry = {
            "timestamp": timestamp,
            "message": message,
            "type": type
        }
        self.thinking_log.append(thinking_entry)
        print(f"[{timestamp}] {type.upper()}: {message}")
    
    def _parse_csv(self, csv_text):
        """
        Parse CSV text into structured data (runs locally)
        
        Args:
            csv_text (str): Raw CSV content
        
        Returns:
            dict: Parsed data with headers and rows
        """
        self._log_thinking("Parsing CSV data locally...")
        
        lines = csv_text.strip().split('\n')
        headers = [header.strip() for header in lines[0].split(',')]
        
        rows = []
        reader = csv.reader(lines[1:], quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for row_data in reader:
            if not row_data:
                continue
                
            # If for some reason we don't have enough columns, pad with empty strings
            while len(row_data) < len(headers):
                row_data.append('')
                
            # Create an object with header keys
            row_object = {}
            for i, header in enumerate(headers):
                if i < len(row_data):
                    row_object[header] = row_data[i].strip()
                else:
                    row_object[header] = ''
                    
            rows.append(row_object)
            
        self._log_thinking(f"Successfully parsed CSV with {len(rows)} rows and {len(headers)} columns")
        return {"headers": headers, "rows": rows}
    
    def analyze(self):
        """
        Perform initial analysis of the CSV data
        
        Returns:
            dict: Analysis results
        """
        self._log_thinking("Starting CSV analysis process...", "plan")
        
        # Extract key information locally using GPT
        headers = self.parsed_data["headers"]
        row_count = len(self.parsed_data["rows"])
        
        # 1. Generate a summary using GPT (local)
        self._log_thinking("Generating summary using GPT...", "execute")
        summary = self._generate_local_summary()
        
        # 2. Identify column types (local)
        self._log_thinking("Identifying column data types...", "execute")
        column_types = self._identify_column_types()
        
        # 3. Generate insights (local)
        self._log_thinking("Generating data insights...", "execute")
        insights = self._generate_data_insights()
        
        # 4. Extract sample data (minimal, non-sensitive)
        self._log_thinking("Extracting minimal sample data...", "execute")
        sample_data = self._get_sample_data(CSV_SAMPLE_SIZE)
        
        # Store analysis results locally
        self.analysis_results = {
            "headers": headers,
            "rowCount": row_count,
            "summary": summary,
            "columnTypes": column_types,
            "sampleData": sample_data,
            "insights": insights
        }
        
        self._log_thinking("Initial analysis complete!", "insight")
        
        return self.analysis_results
    
    def _generate_local_summary(self):
        """
        Generate a summary of the CSV data using GPT (locally)
        
        Returns:
            str: Summary of the data
        """
        try:
            # Get basic data statistics
            headers = self.parsed_data["headers"]
            row_count = len(self.parsed_data["rows"])
            
            # Create a sample to send to OpenAI (first few rows)
            sample_data = self.parsed_data["rows"][:CSV_SAMPLE_SIZE]
            
            # Create a prompt for OpenAI
            prompt = f"""
            I have a CSV dataset with {row_count} rows and the following columns: {', '.join(headers)}
            
            Here's a sample of the first few rows:
            {json.dumps(sample_data, indent=2)}
            
            Provide a concise summary of what this dataset appears to be about, what domain it might belong to,
            and what kind of analysis it might be suitable for. Keep the summary to 3-4 sentences.
            """
            
            self._log_thinking("Calling GPT to summarize data (sharing only minimal sample)...", "execute")
            
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=OPENAI_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that provides concise summaries of datasets."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.choices[0].message.content.strip()
            self._log_thinking(f"Generated summary: {summary}", "insight")
            return summary
            
        except Exception as e:
            self._log_thinking(f"Error calling OpenAI API: {e}", "error")
            # Fallback to simulated response
            return self._simulate_summary()
    
    def _simulate_summary(self):
        """
        Simulate a summary without calling external APIs
        
        Returns:
            str: Simulated summary
        """
        # Look at headers to guess what the data might be about
        headers = self.parsed_data["headers"]
        row_count = len(self.parsed_data["rows"])
        
        # Default values
        data_type = "tabular data"
        domain = "data analytics"
        analysis_type = "exploratory data analysis"
        
        # Check for common patterns in headers to identify the data type
        header_string = " ".join(headers).lower()
        
        # Get a sample of values to help with identification
        sample_values = []
        for row in self.parsed_data["rows"][:5]:
            for header in headers:
                if row[header]:
                    sample_values.append(row[header].lower())
        
        sample_string = " ".join(sample_values)
        combined_text = header_string + " " + sample_string
        
        # Pattern matching logic (simplified)
        if any(term in header_string for term in ["name", "email", "phone", "customer"]):
            data_type = "contact information"
            domain = "customer relationship management"
            
        elif any(term in header_string for term in ["price", "cost", "revenue"]):
            data_type = "financial data"
            domain = "business analytics"
            
        # Generate a simulated summary
        summary = f"This dataset appears to be {data_type} in the domain of {domain}. It contains {row_count} records with {len(headers)} columns: {', '.join(headers)}. The data seems suitable for analysis related to {domain}."
        
        return summary
    
    def _identify_column_types(self):
        """
        Identify the data type of each column (runs locally)
        
        Returns:
            dict: Column types
        """
        column_types = {}
        
        for header in self.parsed_data["headers"]:
            values = [row[header] for row in self.parsed_data["rows"] if row[header]]
            
            if not values:
                column_types[header] = "empty"
                continue
                
            # Check if numeric
            numeric_count = sum(1 for val in values if val.replace('.', '', 1).isdigit() or (val.startswith('-') and val[1:].replace('.', '', 1).isdigit()))
            if numeric_count / len(values) > 0.9:
                column_types[header] = "numeric"
                continue
                
            # Check if date (simple pattern)
            import re
            date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$|^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')
            date_count = sum(1 for val in values if date_pattern.match(val))
            if date_count / len(values) > 0.9:
                column_types[header] = "date"
                continue
                
            # Check if boolean
            bool_pattern = re.compile(r'^(true|false|yes|no|1|0)$', re.IGNORECASE)
            bool_count = sum(1 for val in values if bool_pattern.match(val))
            if bool_count / len(values) > 0.9:
                column_types[header] = "boolean"
                continue
                
            # Default to string
            column_types[header] = "string"
                
        return column_types
    
    def _get_sample_data(self, sample_size=5):
        """
        Get a minimized sample of the data for display
        
        Args:
            sample_size (int): Number of rows to include in sample
        
        Returns:
            list: Sample data
        """
        return self.parsed_data["rows"][:sample_size]
    
    def _generate_data_insights(self, missing_threshold=10):
        """
        Generate insights about the data (runs locally)
        
        Args:
            missing_threshold (int): Percentage threshold for reporting missing values
        
        Returns:
            list: Insights
        """
        insights = []
        
        # Check for missing values
        for header in self.parsed_data["headers"]:
            missing_count = sum(1 for row in self.parsed_data["rows"] if not row[header])
            missing_percentage = (missing_count / len(self.parsed_data["rows"])) * 100
            
            if missing_percentage > missing_threshold:
                insights.append(f'Column "{header}" has {missing_percentage:.1f}% missing values.')
        
        # Check for numeric columns and provide basic stats
        for header in self.parsed_data["headers"]:
            values = []
            for row in self.parsed_data["rows"]:
                try:
                    if row[header] and row[header].replace('.', '', 1).isdigit() or (row[header].startswith('-') and row[header][1:].replace('.', '', 1).isdigit()):
                        values.append(float(row[header]))
                except (ValueError, AttributeError):
                    continue
                    
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                
                insights.append(f'Column "{header}" has numeric values ranging from {min_val} to {max_val} with an average of {avg:.2f}.')
        
        # Identify potential categories in string columns
        for header in self.parsed_data["headers"]:
            values = [row[header] for row in self.parsed_data["rows"] if row[header]]
            
            unique_values = set(values)
            
            if 1 < len(unique_values) <= 10 and len(unique_values) < len(self.parsed_data["rows"]) / 3:
                insights.append(f'Column "{header}" contains {len(unique_values)} distinct categories: {", ".join(unique_values)}.')
        
        return insights
    
    def process_query_with_portia(self, user_message):
        """
        Process user query using Portia and local data
        
        Args:
            user_message (str): User's message/query
        
        Returns:
            dict: Response with visualization data
        """
        if not self.analysis_results:
            self._log_thinking("Analysis results not available. Running analysis first...", "plan")
            self.analyze()
        
        self._log_thinking(f"Processing query with Portia: {user_message}", "plan")
        
        # Use Portia if available, otherwise fall back to local implementation
        if PORTIA_AVAILABLE and self.portia:
            try:
                # Create enhanced query with CSV context
                enhanced_query = f"""
                Answer this question about the CSV data: {user_message}
                
                CSV Summary: {self.analysis_results['summary']}
                
                The CSV has {self.analysis_results['rowCount']} rows and includes these columns: 
                {', '.join(self.analysis_results['headers'])}
                
                Column types: {json.dumps(self.analysis_results['columnTypes'])}
                
                Insights: {json.dumps(self.analysis_results['insights'])}
                """
                
                # Create a simple plan for Portia following the pattern from research_agent.py
                plan = Plan(
                    plan_context={
                        "query": user_message,
                        "csv_data": {
                            "summary": self.analysis_results["summary"],
                            "row_count": self.analysis_results["rowCount"],
                            "headers": self.analysis_results["headers"]
                        }
                    },
                    steps=[
                        {
                            "task": "Understand query intent",
                            "output": "$query_intent",
                            "description": "Determine what aspect of the data the user is asking about"
                        },
                        {
                            "task": "Retrieve data",
                            "output": "$relevant_data",
                            "description": "Get the relevant data and insights from the CSV analysis",
                            "depends_on": ["$query_intent"]
                        },
                        {
                            "task": "Format response",
                            "output": "$final_response",
                            "description": "Create a clear, informative response based on the data",
                            "depends_on": ["$relevant_data"]
                        }
                    ]
                )
                
                # Log plan creation
                self._log_thinking(f"Created Portia plan with {len(plan.steps)} steps", "plan")
                
                # Create a plan run
                plan_run = self.portia.create_plan_run(plan)
                
                # Execute the plan by running a command on Portia
                self._log_thinking("Executing Portia plan...", "execute")
                
                # Set the input for the plan run (the enhanced query)
                plan_run.inputs = {"input": enhanced_query}
                
                # Resume/execute the plan run
                executed_plan_run = self.portia.resume(plan_run_id=plan_run.id)
                
                # Get the final output
                if executed_plan_run.state == PlanRunState.COMPLETE:
                    response = executed_plan_run.outputs.output
                    if not response:
                        # Try to get the output from the last step
                        if executed_plan_run.outputs.step_outputs.get("$final_response"):
                            response = executed_plan_run.outputs.step_outputs.get("$final_response").value
                        else:
                            # Fallback response
                            response = "I've analyzed your data but couldn't generate a complete response."
                else:
                    # If the plan run didn't complete, use a fallback response
                    response = f"Plan execution ended with state: {executed_plan_run.state}. Unable to complete the analysis."
                
                # Collect step information for visualization
                steps = []
                for step in executed_plan_run.steps:
                    self._log_thinking(f"Step {step.index}: {step.task} - {step.state}", "execute")
                    steps.append({
                        "name": step.task,
                        "description": step.description if hasattr(step, "description") else "",
                        "status": step.state.value if hasattr(step.state, "value") else str(step.state)
                    })
                
                # Prepare visualization data
                portia_visualization = {
                    "plan": {
                        "name": "CSV Analysis with Portia",
                        "description": f"Answer: {user_message}"
                    },
                    "steps": steps
                }
                
                # Return results
                return {
                    "response": response,
                    "thinkingVisualization": self.thinking_log,
                    "portiaVisualization": portia_visualization
                }
                
            except Exception as e:
                self._log_thinking(f"Error using Portia: {str(e)}", "error")
                # Continue to fallback
        
        # Fallback to local implementation
        self._log_thinking("Using local implementation", "plan")
        
        # Create manual plan for visualization
        steps = [
            {"name": "Understand Query Intent", "description": "Determine query intent", "status": "in_progress"},
            {"name": "Retrieve Data", "description": "Get relevant data", "status": "not_started"},
            {"name": "Format Response", "description": "Create response", "status": "not_started"}
        ]
        
        # Execute plan steps manually
        query_intent = self._extract_query_intent(user_message)
        steps[0]["status"] = "completed"
        self._log_thinking(f"Query intent identified: {query_intent}", "insight")
        
        steps[1]["status"] = "in_progress"
        relevant_data = self._get_relevant_data_for_intent(query_intent)
        steps[1]["status"] = "completed"
        
        steps[2]["status"] = "in_progress"
        response = self._format_response_for_intent(query_intent, relevant_data)
        steps[2]["status"] = "completed"
        
        return {
            "response": response,
            "thinkingVisualization": self.thinking_log,
            "portiaVisualization": {
                "plan": {
                    "name": "CSV Analysis Plan (Local)",
                    "description": f"Analyze CSV data to answer: {user_message}"
                },
                "steps": steps
            }
        }
    
    def _extract_query_intent(self, user_message):
        """
        Extract the intent of a user query using GPT (locally)
        
        Args:
            user_message (str): User's message
        
        Returns:
            str: Extracted intent
        """
        self._log_thinking("Extracting query intent using GPT...", "execute")
        
        try:
            # Create a prompt for OpenAI
            prompt = f"""
            A user has asked the following question about a CSV dataset:
            "{user_message}"
            
            Classify the intent of this query into one of these categories:
            1. Summary/Overview
            2. Column Information
            3. Data Insights/Patterns
            4. Sample Data Request
            5. Missing Value Analysis
            6. Statistical Analysis
            7. Other
            
            Return only the category name.
            """
            
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=20,
                messages=[
                    {"role": "system", "content": "You are a query intent classifier."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            intent = response.choices[0].message.content.strip()
            return intent
            
        except Exception as e:
            self._log_thinking(f"Error extracting query intent: {e}", "error")
            # Simple fallback classification
            query_lower = user_message.lower()
            
            if any(term in query_lower for term in ["summary", "overview", "about"]):
                return "Summary/Overview"
            elif any(term in query_lower for term in ["column", "field", "headers"]):
                return "Column Information"
            elif any(term in query_lower for term in ["insight", "pattern", "find"]):
                return "Data Insights/Patterns"
            elif any(term in query_lower for term in ["sample", "example", "preview"]):
                return "Sample Data Request"
            elif any(term in query_lower for term in ["miss", "empty", "null"]):
                return "Missing Value Analysis"
            elif any(term in query_lower for term in ["stat", "number", "average", "mean"]):
                return "Statistical Analysis"
            else:
                return "Other"
    
    def _get_relevant_data_for_intent(self, intent):
        """
        Get relevant data based on the query intent
        
        Args:
            intent (str): Query intent
        
        Returns:
            dict: Relevant data
        """
        relevant_data = {}
        
        if intent == "Summary/Overview":
            relevant_data = {
                "summary": self.analysis_results["summary"],
                "rowCount": self.analysis_results["rowCount"],
                "columnCount": len(self.analysis_results["headers"]),
                "headers": self.analysis_results["headers"]
            }
        
        elif intent == "Column Information":
            relevant_data = {
                "headers": self.analysis_results["headers"],
                "columnTypes": self.analysis_results["columnTypes"]
            }
        
        elif intent == "Data Insights/Patterns":
            relevant_data = {
                "insights": self.analysis_results["insights"]
            }
        
        elif intent == "Sample Data Request":
            relevant_data = {
                "headers": self.analysis_results["headers"],
                "sampleData": self.analysis_results["sampleData"],
                "rowCount": self.analysis_results["rowCount"]
            }
        
        elif intent == "Missing Value Analysis":
            relevant_data = {
                "insights": [i for i in self.analysis_results["insights"] if "missing" in i]
            }
        
        elif intent == "Statistical Analysis":
            relevant_data = {
                "insights": [i for i in self.analysis_results["insights"] if "ranging" in i or "average" in i]
            }
        
        else:
            # For general query, include all data
            relevant_data = self.analysis_results
        
        return relevant_data
    
    def _format_response_for_intent(self, intent, relevant_data):
        """
        Format a response based on query intent and relevant data
        
        Args:
            intent (str): Query intent
            relevant_data (dict): Relevant data for the response
        
        Returns:
            str: Formatted response
        """
        if intent == "Summary/Overview":
            return f"{relevant_data['summary']}\n\nThe data contains {relevant_data['rowCount']} rows and {relevant_data['columnCount']} columns. Key columns include: {', '.join(relevant_data['headers'])}."
        
        elif intent == "Column Information":
            column_info = ", ".join([f"{col} ({type_})" for col, type_ in relevant_data["columnTypes"].items()])
            return f"The dataset contains {len(relevant_data['headers'])} columns: {column_info}"
        
        elif intent == "Data Insights/Patterns":
            if relevant_data["insights"]:
                return f"I've analyzed your data and found {len(relevant_data['insights'])} key insights:\n\n{chr(10).join(relevant_data['insights'])}"
            else:
                return "I've analyzed your data but didn't find any significant patterns or insights that stand out. The data appears quite regular."
        
        elif intent == "Sample Data Request":
            headers = ", ".join(relevant_data["headers"])
            separator = "-" * 50
            sample_rows = "\n".join([", ".join([row[h] for h in relevant_data["headers"]]) for row in relevant_data["sampleData"]])
            return f"Here's a preview of your data (showing {len(relevant_data['sampleData'])} of {relevant_data['rowCount']} rows):\n\n{headers}\n{separator}\n{sample_rows}"
        
        elif intent == "Missing Value Analysis":
            if relevant_data["insights"]:
                return f"I checked your data for missing values and found:\n\n{chr(10).join(relevant_data['insights'])}"
            else:
                return "I've analyzed your data for missing values and didn't find any significant gaps. The data appears to be quite complete."
        
        elif intent == "Statistical Analysis":
            if relevant_data["insights"]:
                return f"Here are the statistical highlights from your data:\n\n{chr(10).join(relevant_data['insights'])}"
            else:
                return "I wasn't able to find clear numerical columns to provide statistics on. The data may be primarily categorical or text-based."
        
        else:
            # Default comprehensive response
            all_data = self.analysis_results
            key_insights = ""
            if all_data["insights"]:
                key_insights = f"\n\nKey insights:\n{chr(10).join(all_data['insights'][:3])}"
            return f"I've analyzed your CSV data and found that {all_data['summary']}\n\nThe dataset has {all_data['rowCount']} rows and {len(all_data['headers'])} columns.{key_insights}"

def main():
    """Example usage of the Portia CSV analyzer"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python portia_csv_analyzer.py <csv_file>")
        return
    
    csv_file_path = sys.argv[1]
    try:
        with open(csv_file_path, 'r') as file:
            csv_data = file.read()
        
        print(f"Analyzing CSV file: {csv_file_path}")
        analyzer = PortiaCsvAnalyzer(csv_data)
        
        # First question - get general info
        print("\nPerforming initial analysis...")
        analysis = analyzer.analyze()
        
        print("\nSummary:")
        print(analysis["summary"])
        
        print("\nColumn Types:")
        for col, type_ in analysis["columnTypes"].items():
            print(f"- {col}: {type_}")
        
        print("\nInsights:")
        for insight in analysis["insights"]:
            print(f"- {insight}")
        
        # Interactive query mode
        print("\n=== CSV Analyzer Interactive Mode ===")
        print("Ask questions about your CSV data (or type 'exit' to quit):")
        
        # Show Portia status
        if PORTIA_AVAILABLE and analyzer.portia:
            print("✓ Portia integration active - your queries will be processed with Portia plans")
        else:
            print("⚠ Portia integration not available - using fallback implementation")
        
        while True:
            query = input("\n> ")
            if query.lower() in ["exit", "quit"]:
                break
            
            # Process the query
            print("\n📋 Processing query...")
            result = analyzer.process_query_with_portia(query)
            
            # Display the Plan information
            plan = result["portiaVisualization"]["plan"]
            print(f"\n📝 Plan: {plan['name']}")
            print(f"Description: {plan['description']}")
            
            # Display each step in the plan
            print("\n📊 Steps:")
            for i, step in enumerate(result["portiaVisualization"]["steps"]):
                status_icon = "✓" if step["status"] == "completed" or step["status"] == "COMPLETE" else "⏳"
                print(f"{status_icon} Step {i+1}: {step['name']} - {step['status']}")
            
            # Display thinking process logs
            print("\n🧠 Thinking Process:")
            for log in result["thinkingVisualization"][-5:]:  # Show last 5 thinking logs
                print(f"[{log['timestamp']}] {log['type'].upper()}: {log['message']}")
            
            # Display the final response
            print("\n🤖 Response:")
            print(result["response"])
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 