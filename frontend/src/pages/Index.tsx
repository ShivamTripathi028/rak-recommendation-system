
import { Link } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronRight } from "lucide-react";

const Index = () => {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header Section */}
      <header className="bg-blue-gradient py-16 text-white text-center px-4">
        <div className="mx-auto mb-6">
          <div className="w-20 h-20 bg-rak-blue rounded-full flex items-center justify-center mx-auto border-2 border-white">
            <span className="text-3xl font-bold">R</span>
          </div>
        </div>
        <h1 className="text-4xl font-bold mb-4">Welcome to the RAK Help Hub</h1>
        <p className="text-xl max-w-2xl mx-auto">
          Find the support or information you need for your RAKwireless products and solutions.
        </p>
      </header>

      {/* FAQ Section */}
      <div className="bg-white py-8">
        <div className="container mx-auto px-4">
          <div className="border-l-4 border-rak-blue bg-blue-50 p-4 mb-8 rounded-r">
            <h2 className="text-lg font-semibold text-rak-blue">How does this work?</h2>
            <p className="text-gray-600 mt-1">
              Select the type of support you need below, and we'll guide you to the right resources.
            </p>
          </div>

          <h2 className="text-2xl font-bold text-center mb-8">Select Your Request Type</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Technical Support Card */}
            <Card className="p-6 shadow-md hover:shadow-lg transition-shadow duration-300 flex flex-col">
              <div className="mb-4">
                <img 
                  src="https://images.unsplash.com/photo-1488590528505-98d2b5aba04b" 
                  alt="Technical Support" 
                  className="w-full h-40 object-cover rounded-md mb-4"
                />
                <h3 className="text-xl font-semibold text-rak-blue mb-3">Technical Support</h3>
                <p className="text-gray-600 mb-6">
                  Get help with device setup, configuration, firmware issues, troubleshooting, and more.
                </p>
              </div>
              <div className="mt-auto">
                <Button variant="outline" className="w-full justify-between">
                  Request Technical Support
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </Card>

            {/* Sales Inquiry Card */}
            <Card className="p-6 shadow-md hover:shadow-lg transition-shadow duration-300 flex flex-col">
              <div className="mb-4">
                <img 
                  src="https://images.unsplash.com/photo-1518770660439-4636190af475" 
                  alt="Sales Inquiry" 
                  className="w-full h-40 object-cover rounded-md mb-4"
                />
                <h3 className="text-xl font-semibold text-rak-blue mb-3">Sales Inquiry</h3>
                <p className="text-gray-600 mb-6">
                  Need guidance choosing the right product? Fill out our form detailing your use case, region, and application field.
                </p>
              </div>
              <div className="mt-auto">
                <Button variant="outline" className="w-full justify-between">
                  Submit Inquiry
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </Card>

            {/* Product Recommendation Card */}
            <Card className="p-6 shadow-md hover:shadow-lg transition-shadow duration-300 flex flex-col">
              <div className="mb-4">
                <img 
                  src="https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d" 
                  alt="Product Recommendation" 
                  className="w-full h-40 object-cover rounded-md mb-4"
                />
                <h3 className="text-xl font-semibold text-rak-blue mb-3">Product Recommendation</h3>
                <p className="text-gray-600 mb-6">
                  Get personalized IoT product recommendations based on your use case.
                </p>
              </div>
              <div className="mt-auto">
                <Button asChild className="w-full justify-between bg-rak-blue hover:bg-rak-light-blue">
                  <Link to="/recommendation-form">
                    Get Recommendations
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-auto py-6 bg-gray-50 border-t">
        <div className="container mx-auto text-center text-gray-500 text-sm">
          © 2025 RAKwireless Technology Limited. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default Index;
