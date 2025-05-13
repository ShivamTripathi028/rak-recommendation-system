// src/App.tsx
// Removed: import { Toaster } from "@/components/ui/toaster"; 
import { Toaster as SonnerToaster } from "@/components/ui/sonner"; // Using Sonner
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom"; // Removed Navigate if not needed
import NotFound from "./pages/NotFound";
import RecommendationForm from "./pages/RecommendationForm";
import RecommendationResults from "./pages/RecommendationResults";
// If you decide to use your Index.tsx as the landing page later:
// import Index from "./pages/Index"; 

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <SonnerToaster /> {/* Only Sonner, as pages use it */}
      
      <BrowserRouter>
        <Routes>
          {/* Option 1: Form is the homepage (as per your current code) */}
          <Route path="/" element={<RecommendationForm />} />
          
          {/* Option 2: If you want Index.tsx (with the 3 cards) as homepage later: */}
          {/* <Route path="/" element={<Index />} /> */}
          {/* <Route path="/recommendation-form" element={<RecommendationForm />} /> */}

          <Route path="/recommendation-results" element={<RecommendationResults />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;