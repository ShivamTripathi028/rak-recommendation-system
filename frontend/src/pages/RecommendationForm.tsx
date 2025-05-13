// frontend/src/pages/RecommendationForm.tsx
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner"; // Ensure sonner is your toast library

// Define the structure of a single recommendation item from the backend
interface Recommendation {
  product_name: string;
  product_model?: string;
  category?: string;
  score: number;
  reasoning: string;
  similarity_score: number;
}

// Define the structure of the API response for /submit-requirement
interface ApiSubmitResponse {
  recommendations: Recommendation[];
  submission_id: number;
  extracted_requirements: any; // You can define a more specific type if needed
}

const RecommendationForm = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    requirements: "", 
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    toast.dismiss(); 

    if (!formData.name.trim() || !formData.email.trim() || !formData.requirements.trim()) {
      toast.error("Please fill in all fields");
      setIsSubmitting(false);
      return;
    }
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      toast.error("Please enter a valid email address");
      setIsSubmitting(false);
      return;
    }

    try {
      const backendUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
      
      const response = await fetch(`${backendUrl}/submit-requirement`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: formData.name,
            email: formData.email,
            description: formData.requirements // API expects 'description'
        }),
      });

      setIsSubmitting(false);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error fetching recommendations." }));
        console.error("API Error Data:", errorData);
        toast.error(errorData.detail || `Error: ${response.statusText}`);
        return;
      }

      const apiResponse: ApiSubmitResponse = await response.json();
      
      navigate("/recommendation-results", { 
        state: { 
          userName: formData.name,
          userEmail: formData.email, 
          initialRequirements: formData.requirements,
          recommendations: apiResponse.recommendations,
          submissionId: apiResponse.submission_id,
          extractedRequirements: apiResponse.extracted_requirements
        } 
      });

    } catch (error) {
      setIsSubmitting(false);
      console.error("Submission error:", error);
      toast.error(`Failed to get recommendations. ${(error as Error).message || "Please try again."}`);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="bg-blue-gradient py-12 text-white text-center px-4">
        <div className="mx-auto mb-4">
          <div className="w-16 h-16 bg-rak-blue rounded-full flex items-center justify-center mx-auto border-2 border-white">
            <span className="text-2xl font-bold">R</span>
          </div>
        </div>
        <h1 className="text-3xl font-bold mb-2">Product Recommendation Form</h1>
        <p className="text-lg max-w-2xl mx-auto">
          Help us understand your needs for personalized product recommendations
        </p>
      </header>

      <div className="flex-grow bg-gray-50 py-8">
        <div className="container mx-auto px-4 max-w-2xl">
          <Card className="p-6 shadow-md">
            <form onSubmit={handleSubmit}>
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    name="name"
                    placeholder="Enter your name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="Enter your email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="requirements">Requirement Description</Label>
                  <Textarea
                    id="requirements"
                    name="requirements"
                    placeholder="Please describe your use case and requirements (e.g., I need a compact LoRaWAN module...)"
                    rows={5}
                    value={formData.requirements}
                    onChange={handleChange}
                    required
                  />
                </div>
                
                <div className="flex justify-end pt-4">
                  <Button 
                    type="submit"
                    className="bg-rak-blue hover:bg-rak-light-blue"
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? "Processing..." : "Get Recommendations"}
                  </Button>
                </div>
              </div>
            </form>
          </Card>
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

export default RecommendationForm;