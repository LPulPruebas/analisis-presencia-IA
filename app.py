import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
import streamlit as st
import json
import re
from urllib.parse import urlparse
import time

# Configuración de API keys
openai.api_key = "tu_api_key_de_openai"  # Reemplazar con tu API key
ANTHROPIC_API_KEY = "tu_api_key_de_anthropic"  # Reemplazar con tu API key
GEMINI_API_KEY = "tu_api_key_de_gemini"  # Reemplazar con tu API key

class AILinkabilityTool:
    def __init__(self, domain, keywords=None, category=None):
        self.domain = domain
        self.keywords = keywords or []
        self.category = category
        self.metadata = {}
        self.prompts = []
        self.results = {
            "openai": [],
            "anthropic": [],
            "gemini": []
        }
        self.metrics = {
            "inclusion_rate": 0,
            "prominence_score": 0,
            "citation_score": 0,
            "dataset_score": 0
        }
        
    def extract_metadata(self):
        """Extraer metadatos SEO del dominio"""
        url = f"https://{self.domain}" if not self.domain.startswith('http') else self.domain
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraer título
            self.metadata['title'] = soup.title.string if soup.title else ""
            
            # Extraer meta descripción
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            self.metadata['description'] = meta_desc['content'] if meta_desc else ""
            
            # Extraer encabezados
            self.metadata['h1'] = [h1.text.strip() for h1 in soup.find_all('h1')]
            self.metadata['h2'] = [h2.text.strip() for h2 in soup.find_all('h2')]
            
            # Extraer palabras clave si no se proporcionaron
            if not self.keywords:
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords and meta_keywords['content']:
                    self.keywords = [k.strip() for k in meta_keywords['content'].split(',')]
            
            return True
            
        except Exception as e:
            print(f"Error al extraer metadatos: {e}")
            return False
    
    def generate_prompts(self, num_prompts=5):
        """Generar prompts basados en los metadatos y palabras clave"""
        
        templates = [
            "¿Cuáles son las principales empresas en {keyword}?",
            "¿Qué web ofrece las mejores soluciones sobre {keyword}?",
            "¿Cuáles son los mejores recursos para aprender sobre {keyword}?",
            "¿Qué sitios web recomendarías para {keyword}?",
            "¿Quiénes son referentes en {keyword}?",
            "¿Dónde puedo encontrar información confiable sobre {keyword}?",
            "¿Qué empresas destacan en el sector de {keyword}?",
            "Si busco información sobre {keyword}, ¿qué sitios web debería visitar?"
        ]
        
        # Combinar palabras clave con el título y descripción
        all_keywords = self.keywords.copy()
        
        if self.category:
            all_keywords.append(self.category)
            
        if 'title' in self.metadata and self.metadata['title']:
            # Extraer palabras clave del título
            title_words = re.findall(r'\b\w{4,}\b', self.metadata['title'])
            all_keywords.extend(title_words)
            
        # Eliminar duplicados y palabras vacías
        all_keywords = list(set([k for k in all_keywords if len(k) > 3]))
        
        # Generar prompts
        for keyword in all_keywords[:min(len(all_keywords), num_prompts)]:
            template = templates[len(self.prompts) % len(templates)]
            self.prompts.append(template.format(keyword=keyword))
            
        return self.prompts
    
    def query_llm(self, provider, prompt):
        """Consultar a diferentes LLMs"""
        result = {"prompt": prompt, "contains_domain": False, "position": -1, "has_link": False, "description": ""}
        
        try:
            if provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil que responde preguntas con información precisa."},
                        {"role": "user", "content": prompt}
                    ]
                )
                result["response"] = response.choices[0].message.content
                
            elif provider == "anthropic":
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "content-type": "application/json"
                }
                data = {
                    "model": "claude-2",
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": 1000
                }
                response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=data)
                result["response"] = response.json().get("completion", "")
                
            elif provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                data = {
                    "contents": [{"parts":[{"text": prompt}]}]
                }
                response = requests.post(url, json=data)
                result["response"] = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Analizar la respuesta
            self._analyze_response(result)
            return result
            
        except Exception as e:
            print(f"Error al consultar {provider}: {e}")
            result["error"] = str(e)
            return result
    
    def _analyze_response(self, result):
        """Analizar si la respuesta contiene el dominio y cómo aparece"""
        response_text = result["response"]
        domain_name = urlparse(self.domain).netloc if '.' in self.domain else self.domain
        
        # Verificar si contiene el dominio
        if domain_name.lower() in response_text.lower():
            result["contains_domain"] = True
            
            # Encontrar posición relativa (normalizada de 0 a 1)
            position = response_text.lower().find(domain_name.lower())
            result["position"] = position / len(response_text)
            
            # Verificar si contiene un enlace
            link_pattern = rf'https?://(?:www\.)?{re.escape(domain_name)}'
            result["has_link"] = bool(re.search(link_pattern, response_text))
            
            # Extraer descripción contextual (50 caracteres antes y después)
            start = max(0, position - 50)
            end = min(len(response_text), position + len(domain_name) + 50)
            result["description"] = response_text[start:end]
    
    def run_analysis(self):
        """Ejecutar el análisis completo"""
        # Extraer metadatos
        if not self.extract_metadata():
            return {"error": "No se pudieron extraer los metadatos del sitio"}
        
        # Generar prompts
        self.generate_prompts()
        
        # Consultar LLMs
        providers = ["openai", "anthropic", "gemini"]
        total_queries = len(self.prompts) * len(providers)
        query_count = 0
        
        for prompt in self.prompts:
            for provider in providers:
                result = self.query_llm(provider, prompt)
                self.results[provider].append(result)
                query_count += 1
                print(f"Progreso: {query_count}/{total_queries} ({query_count/total_queries*100:.1f}%)")
                # Pausa para evitar límites de rate
                time.sleep(1)
        
        # Calcular métricas
        self.calculate_metrics()
        
        return {
            "domain": self.domain,
            "metadata": self.metadata,
            "prompts": self.prompts,
            "results": self.results,
            "metrics": self.metrics
        }
    
    def calculate_metrics(self):
        """Calcular métricas de rendimiento"""
        all_results = []
        for provider_results in self.results.values():
            all_results.extend(provider_results)
        
        if not all_results:
            return
        
        # Tasa de inclusión
        mentions = [r for r in all_results if r.get("contains_domain")]
        self.metrics["inclusion_rate"] = len(mentions) / len(all_results) if all_results else 0
        
        # Puntuación de prominencia (más bajo es mejor)
        positions = [r.get("position") for r in mentions if r.get("position") >= 0]
        self.metrics["prominence_score"] = sum(positions) / len(positions) if positions else 1
        
        # Puntuación de citas
        citations = [r for r in mentions if r.get("has_link")]
        self.metrics["citation_score"] = len(citations) / len(mentions) if mentions else 0
        
        # Puntuación de dataset (simulada - en producción se consultaría Common Crawl)
        self.metrics["dataset_score"] = 0.7  # Valor simulado
    
    def generate_report(self):
        """Generar informe con recomendaciones"""
        report = {
            "domain": self.domain,
            "analysis_date": time.strftime("%Y-%m-%d"),
            "metrics": self.metrics,
            "mentions": sum(1 for r in self.results["openai"] + self.results["anthropic"] + self.results["gemini"] if r.get("contains_domain")),
            "total_queries": len(self.prompts) * 3,  # 3 proveedores
            "recommendations": []
        }
        
        # Generar recomendaciones
        if self.metrics["inclusion_rate"] < 0.3:
            report["recommendations"].append({
                "type": "critical",
                "title": "Baja tasa de inclusión",
                "description": "Tu dominio rara vez aparece en las respuestas. Considera crear más contenido autorizado sobre tus temas clave."
            })
        
        if self.metrics["prominence_score"] > 0.7:
            report["recommendations"].append({
                "type": "warning",
                "title": "Baja prominencia",
                "description": "Cuando tu sitio aparece, lo hace muy tarde en las respuestas. Mejora tu autoridad temática."
            })
        
        if self.metrics["citation_score"] < 0.2:
            report["recommendations"].append({
                "type": "improvement",
                "title": "Pocas citas con enlaces",
                "description": "Pocos LLMs enlazan directamente a tu sitio. Considera estrategias para aumentar backlinks de calidad."
            })
        
        return report

# Interfaz de Streamlit
def create_streamlit_app():
    st.title("🤖 AI Linkability Analyzer")
    st.write("Analiza cómo los LLMs mencionan tu sitio web en sus respuestas")
    
    with st.form("analysis_form"):
        domain = st.text_input("Dominio web", placeholder="ejemplo.com")
        keywords = st.text_area("Palabras clave (opcional, una por línea)")
        category = st.text_input("Categoría del negocio (opcional)", placeholder="sostenibilidad en construcción")
        
        submitted = st.form_submit_button("Analizar")
        
        if submitted and domain:
            keywords_list = [k.strip() for k in keywords.split('\n') if k.strip()]
            
            with st.spinner("Analizando... esto puede tomar unos minutos"):
                tool = AILinkabilityTool(domain, keywords_list, category)
                results = tool.run_analysis()
                report = tool.generate_report()
                
                st.success("¡Análisis completado!")
                
                # Mostrar métricas
                st.header("Métricas de rendimiento")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tasa de inclusión", f"{report['metrics']['inclusion_rate']:.1%}")
                col2.metric("Prominencia", f"{report['metrics']['prominence_score']:.2f}")
                col3.metric("Citas con enlaces", f"{report['metrics']['citation_score']:.1%}")
                col4.metric("Score en datasets", f"{report['metrics']['dataset_score']:.1%}")
                
                # Mostrar menciones
                st.header("Menciones detectadas")
                st.write(f"Tu dominio fue mencionado en {report['mentions']} de {report['total_queries']} consultas.")
                
                # Mostrar recomendaciones
                st.header("Recomendaciones")
                for rec in report["recommendations"]:
                    if rec["type"] == "critical":
                        st.error(f"**{rec['title']}**: {rec['description']}")
                    elif rec["type"] == "warning":
                        st.warning(f"**{rec['title']}**: {rec['description']}")
                    else:
                        st.info(f"**{rec['title']}**: {rec['description']}")
                
                # Mostrar ejemplos de menciones
                st.header("Ejemplos de menciones")
                mentions = []
                for provider, results in tool.results.items():
                    for result in results:
                        if result.get("contains_domain"):
                            mentions.append({
                                "provider": provider.capitalize(),
                                "prompt": result["prompt"],
                                "description": result["description"],
                                "has_link": "Sí" if result["has_link"] else "No"
                            })
                
                if mentions:
                    st.dataframe(pd.DataFrame(mentions[:10]))  # Mostrar primeras 10 menciones
                else:
                    st.write("No se encontraron menciones de tu dominio.")
                
                # Permitir descargar el informe completo
                st.download_button(
                    "Descargar informe completo (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"linkability_report_{domain}_{time.strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    create_streamlit_app()
