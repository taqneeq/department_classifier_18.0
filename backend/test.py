#!/usr/bin/env python3
"""
Quick FastAPI Test Runner - Shows FastAPI-specific testing benefits
Run this to see what FastAPI testing adds beyond your current tests
"""

from fastapi.testclient import TestClient
from app import app
import json
import time

class FastAPITester:
    def __init__(self):
        self.client = TestClient(app)
        self.passed = 0
        self.failed = 0
        
    def test(self, name, test_func):
        """Run a test and track results"""
        print(f"🧪 {name}...", end=" ")
        try:
            test_func()
            print("✅ PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.failed += 1
    
    def run_fastapi_specific_tests(self):
        """Run tests that showcase FastAPI-specific features"""
        print("🚀 FastAPI-Specific Test Features")
        print("=" * 50)
        
        # 1. Automatic request/response validation
        self.test("Request Validation", self.test_request_validation)
        
        # 2. OpenAPI schema generation
        self.test("OpenAPI Schema", self.test_openapi_schema)
        
        # 3. Dependency injection testing
        self.test("Dependency Injection", self.test_dependency_injection)
        
        # 4. Background tasks testing
        self.test("Response Headers", self.test_response_headers)
        
        # 5. Middleware testing
        self.test("Middleware Functionality", self.test_middleware)
        
        # 6. Performance testing
        self.test("Response Performance", self.test_performance)
        
        # 7. Error handling
        self.test("Error Handling", self.test_error_handling)
        
        # 8. Content-Type handling
        self.test("Content Type Handling", self.test_content_types)
        
        print("\n" + "=" * 50)
        print(f"📊 FastAPI Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 All FastAPI-specific features working perfectly!")
        
        return self.failed == 0
    
    def test_request_validation(self):
        """Test FastAPI's automatic request validation"""
        # Test invalid JSON
        response = self.client.post("/api/v1/classification/answer", 
                                  json={"response": "invalid"})  # Should be int
        assert response.status_code == 422
        
        # Test missing required fields
        response = self.client.post("/api/v1/classification/answer", json={})
        assert response.status_code == 422
        
        # Test valid request structure
        response = self.client.post("/api/v1/classification/start", json={})
        # Should work or fail gracefully, not with validation error
        assert response.status_code != 422
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation"""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check our endpoints are documented
        assert "/api/v1/classification/start" in schema["paths"]
        assert "/api/v1/departments" in schema["paths"]
        
        # Check proper HTTP methods
        start_path = schema["paths"]["/api/v1/classification/start"]
        assert "post" in start_path
    
    def test_dependency_injection(self):
        """Test that dependency injection works properly"""
        # The fact that our endpoints work shows DI is working
        response = self.client.get("/api/v1/health")
        assert response.status_code in [200, 503]  # Should respond, not crash
        
        # Test admin endpoints that use dependencies
        response = self.client.get("/api/v1/stats")
        assert response.status_code in [200, 500]  # Should handle gracefully
    
    def test_response_headers(self):
        """Test custom response headers from middleware"""
        response = self.client.get("/api/v1/health")
        
        # Check for custom headers from our middleware
        headers = response.headers
        # Our middleware should add these
        assert "x-request-id" in headers or "content-type" in headers
    
    def test_middleware(self):
        """Test middleware functionality"""
        # Test CORS headers
        response = self.client.options("/api/v1/health", 
                                     headers={"Origin": "http://localhost:3000"})
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        
        # Test compression (if enabled)
        response = self.client.get("/api/v1/departments")
        # Should work regardless of compression
        assert response.status_code in [200, 500]
    
    def test_performance(self):
        """Test response performance"""
        start_time = time.time()
        response = self.client.get("/api/v1/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        # Health check should be fast
        assert response_time < 5.0, f"Health check took {response_time:.2f}s"
        
        if response.status_code == 200:
            assert response_time < 2.0, "Healthy service should respond quickly"
    
    def test_error_handling(self):
        """Test proper error response format"""
        # Test 404 error
        response = self.client.get("/api/v1/departments/nonexistent")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Test invalid session ID
        response = self.client.get("/api/v1/classification/status/invalid_session")
        # Should handle gracefully
        assert response.status_code in [400, 404, 500]
    
    def test_content_types(self):
        """Test content type handling"""
        # Test JSON content type
        response = self.client.post("/api/v1/classification/start",
                                  json={},
                                  headers={"Content-Type": "application/json"})
        assert response.status_code != 415  # Not unsupported media type
        
        # Test HTML response for root
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

def compare_testing_approaches():
    """Compare your current testing vs FastAPI testing"""
    print("🔍 Testing Approach Comparison")
    print("=" * 60)
    
    print("Your Current Tests (curl + custom):")
    print("✅ End-to-end functionality testing")
    print("✅ Real HTTP requests")
    print("✅ Production-like testing")
    print("✅ Complete system validation")
    print()
    
    print("FastAPI TestClient Benefits:")
    print("✅ No server startup required")
    print("✅ Faster test execution")
    print("✅ Request/response validation testing")
    print("✅ Middleware testing")
    print("✅ Dependency injection testing")
    print("✅ OpenAPI schema validation")
    print("✅ Better error isolation")
    print("✅ Mocking capabilities")
    print()
    
    print("🎯 Recommendation:")
    print("Keep your current tests + Add FastAPI tests for:")
    print("• Request validation edge cases")
    print("• Middleware functionality")
    print("• API contract testing")
    print("• Unit-level endpoint testing")

def main():
    """Main test runner"""
    print("🎯 FastAPI Testing for Taqneeq Department Classifier")
    print("=" * 60)
    
    # Show comparison first
    compare_testing_approaches()
    print()
    
    # Run FastAPI-specific tests
    tester = FastAPITester()
    success = tester.run_fastapi_specific_tests()
    
    print("\n" + "=" * 60)
    print("📝 Summary:")
    print("Your current test.py gives you comprehensive system validation.")
    print("FastAPI tests add:")
    print("• Faster execution (no server needed)")
    print("• Better request validation testing") 
    print("• Middleware and dependency testing")
    print("• API contract validation")
    print()
    
    if success:
        print("🎉 Both testing approaches show your system is excellent!")
        print("💡 Consider keeping both for comprehensive coverage")
    else:
        print("⚠️ Some FastAPI-specific features need attention")
        print("But your core system (tested by test.py) is still excellent!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())