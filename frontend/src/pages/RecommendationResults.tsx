// frontend/src/pages/RecommendationResults.tsx
import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronRight, ExternalLink } from "lucide-react";
import { toast } from "sonner";

interface Recommendation {
  product_name: string;
  product_model?: string;
  category?: string;
  score: number;
  reasoning: string;
  similarity_score: number;
}

interface LocationState {
  userName: string;
  userEmail: string;
  initialRequirements: string;
  recommendations: Recommendation[];
  submissionId: number;
  extractedRequirements?: any; // Optional, for display/debug
}

const RecommendationResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const state = location.state as LocationState | null;

  const { 
    userName = "User", 
    userEmail = "user@example.com", // Make sure this is passed from form
    initialRequirements = "No requirements were provided.", 
    recommendations = [], 
    submissionId = 0, // Default to 0 if not provided
    // extractedRequirements // Can be used for display
  } = state || {};
  
  const [selectedProducts, setSelectedProducts] = useState<Recommendation[]>([]);
  const [isLogging, setIsLogging] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
    if (!state || !state.recommendations || (Array.isArray(state.recommendations) && state.recommendations.length === 0 && state.initialRequirements)) {
        if (state && Array.isArray(state.recommendations) && state.recommendations.length === 0) {
            toast.info("No specific products matched your query. Try rephrasing or contact sales.");
        } else if (!state) {
            toast.error("No recommendation data found. Please submit the form again.");
            navigate("/recommendation-form"); 
        }
    }
  }, [state, navigate]);


  const handleCheckboxChange = (product: Recommendation) => {
    setSelectedProducts(prev => {
      const isSelected = prev.some(p => 
        p.product_name === product.product_name && p.product_model === product.product_model
      );
      if (isSelected) {
        return prev.filter(p => !(p.product_name === product.product_name && p.product_model === product.product_model));
      } else {
        return [...prev, product];
      }
    });
  };

  const handleGoToStore = async () => {
    if (selectedProducts.length === 0) {
      toast.info("No products selected. Redirecting to store homepage...");
      window.location.href = import.meta.env.VITE_STORE_BASE_URL || "https://store.rakwireless.com";
      return;
    }
    
    setIsLogging(true);
    toast.dismiss();

    const productsToLog = selectedProducts.map(p => ({
        name: p.product_name,
        model: p.product_model || "" 
    }));

    try {
      const backendUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${backendUrl}/log-selection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            submission_id: submissionId,
            name: userName,
            email: userEmail, 
            selected_products: productsToLog
        }),
      });

      setIsLogging(false);
      const responseData = await response.json().catch(() => null); // Try to parse JSON, but handle if not

      if (!response.ok) {
        toast.error(responseData?.detail || "Failed to log selection. Redirecting anyway.");
      } else {
        toast.success("Your selections have been noted!");
        window.location.href = responseData?.redirect_url || import.meta.env.VITE_STORE_BASE_URL || "https://store.rakwireless.com";
        return; 
      }
    } catch (error) {
      setIsLogging(false);
      console.error("Logging selection error:", error);
      toast.error(`Error logging selection. ${(error as Error).message || "Redirecting anyway..."}`);
    }
    // Fallback redirect
    window.location.href = import.meta.env.VITE_STORE_BASE_URL || "https://store.rakwireless.com";
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="bg-blue-gradient py-12 text-white text-center px-4">
        <div className="mx-auto mb-4">
          <div className="w-16 h-16 bg-rak-blue rounded-full flex items-center justify-center mx-auto border-2 border-white">
            <span className="text-2xl font-bold">R</span>
          </div>
        </div>
        <h1 className="text-3xl font-bold mb-2">Your Personalized Recommendations</h1>
        <p className="text-lg max-w-2xl mx-auto">
          Hi {userName}, based on your requirements, we recommend the following:
        </p>
      </header>

      <div className="flex-grow bg-gray-50 py-8">
        <div className="container mx-auto px-4">
          <div className="bg-white p-5 rounded-lg shadow mb-8 max-w-3xl mx-auto">
            <p className="text-gray-700 mt-1 text-sm">
              <strong>Your Requirement:</strong> <span className="italic">"{initialRequirements.substring(0, 200)}{initialRequirements.length > 200 ? '...' : ''}"</span>
            </p>
          </div>
          
          {recommendations && recommendations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
              {recommendations.map((product, index) => (
                <Card key={`${product.product_name}-${product.product_model || index}`} className="p-5 shadow-md hover:shadow-lg transition-shadow duration-300 flex flex-col">
                  <h3 className="text-xl font-semibold text-rak-blue mb-1">
                    {product.product_name}
                  </h3>
                  {product.product_model && (
                     <p className="text-xs text-gray-500 mb-2">Model: {product.product_model}</p>
                  )}
                  <p className="text-sm text-gray-700 mb-1"><strong>Category:</strong> {product.category || 'N/A'}</p>
                  <p className="text-sm text-gray-700 mb-1"><strong>Relevance Score:</strong> {product.score.toFixed(0)}</p>
                  <p className="text-sm text-gray-700 mb-1"><strong>Detail Similarity:</strong> {product.similarity_score.toFixed(2)}</p>
                  <p className="text-xs text-gray-500 mb-3 h-24 overflow-y-auto p-1 border rounded bg-gray-50">
                    <strong>Reasoning:</strong> {product.reasoning}
                  </p>
                  
                  <div className="flex items-center space-x-2 mt-auto pt-3 border-t">
                    <Checkbox 
                      id={`product-${product.product_name.replace(/\s+/g, '-')}-${index}`} 
                      checked={selectedProducts.some(p => p.product_name === product.product_name && p.product_model === product.product_model)}
                      onCheckedChange={() => handleCheckboxChange(product)}
                    />
                    <label 
                      htmlFor={`product-${product.product_name.replace(/\s+/g, '-')}-${index}`} 
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                    >
                      Select this product
                    </label>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="p-6 shadow-md text-center max-w-xl mx-auto">
                <h3 className="text-xl font-semibold text-rak-blue mb-3">No Specific Recommendations Found</h3>
                <p className="text-gray-600">
                    We couldn't find specific products based on the details provided. 
                    Please try rephrasing your requirement or contact our sales team for assistance.
                </p>
            </Card>
          )}
          
          {recommendations && recommendations.length > 0 && (
            <div className="flex justify-center mt-10">
              <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md">
                <Button variant="outline" className="flex-1" onClick={() => navigate("/recommendation-form")}>
                  New Recommendation
                </Button>
                <Button 
                  className="bg-rak-blue hover:bg-rak-light-blue flex-1"
                  onClick={handleGoToStore}
                  disabled={isLogging || selectedProducts.length === 0}
                >
                  {isLogging ? "Processing..." : "Proceed to Store with Selection"}
                  {selectedProducts.length > 0 && <ExternalLink className="ml-2 h-4 w-4" />}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      <footer className="py-6 bg-gray-50 border-t">
        <div className="container mx-auto text-center text-gray-500 text-sm">
          © 2025 RAKwireless Technology Limited. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default RecommendationResults;