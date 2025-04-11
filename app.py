import requests
from bs4 import BeautifulSoup
import openai # Importante: Mantener el import aunque la key se asigne después
import pandas as pd
import streamlit as st
import json
import re
from urllib.parse import urlparse
import time
import os # Necesario para una comprobación opcional

# --- Configuración de API keys usando st.secrets ---
# Streamlit buscará estas claves en `.streamlit/secrets.toml` si corres localmente,
# o en los secretos configurados en Streamlit Cloud si está desplegado.

# Es buena práctica verificar si los secretos existen, especialmente para dar mensajes útiles
# si el usuario no los ha configurado.

missing_secrets = []
try:
    openai_api_key = st.secrets["api_keys"]["openai"]
    openai.api_key = openai_api_key # Asignar la clave a la librería openai
except KeyError:
    missing_secrets.append("OpenAI")
    openai_api_key = None # O manejar el error como prefieras

try:
    ANTHROPIC_API_KEY = st.secrets["api_keys"]["anthropic"]
except KeyError:
    missing_secrets.append("Anthropic")
    ANTHROPIC_API_KEY = None

try:
    GEMINI_API_KEY = st.secrets["api_keys"]["gemini"]
except KeyError:
    missing_secrets.append("Gemini")
    GEMINI_API_KEY = None

# Opcional: Mostrar un mensaje de advertencia si faltan claves
if missing_secrets:
    missing_keys_str = ", ".join(missing_secrets)
    st.warning(f"Advertencia: No se encontraron claves API para {missing_keys_str} en la configuración de secretos. Las funciones relacionadas con estos servicios podrían no funcionar.")
    # Podrías incluso detener la ejecución si alguna clave es esencial:
    # if "OpenAI" in missing_secrets: # Ejemplo
    #     st.error("La clave API de OpenAI es necesaria para continuar. Por favor, configúrala.")
    #     st.stop()

# --- Resto de tu clase y código (sin cambios en la lógica interna) ---

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
            response.raise_for_status() # Verifica errores HTTP
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extraer título
            self.metadata['title'] = soup.title.string if soup.title else ""

            # Extraer meta descripción
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            self.metadata['description'] = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else ""

            # Extraer encabezados
            self.metadata['h1'] = [h1.text.strip() for h1 in soup.find_all('h1')]
            self.metadata['h2'] = [h2.text.strip() for h2 in soup.find_all('h2')]

            # Extraer palabras clave si no se proporcionaron
            if not self.keywords:
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords and 'content' in meta_keywords.attrs and meta_keywords['content']:
                    self.keywords = [k.strip() for k in meta_keywords['content'].split(',')]

            # Si aún no hay keywords, intentar extraer del título o descripción
            if not self.keywords and self.metadata.get('title'):
                 title_words = re.findall(r'\b\w{4,}\b', self.metadata['title'].lower())
                 self.keywords.extend(title_words)
            if not self.keywords and self.metadata.get('description'):
                 desc_words = re.findall(r'\b\w{4,}\b', self.metadata['description'].lower())
                 self.keywords.extend(desc_words)

            # Limpiar y eliminar duplicados
            self.keywords = list(set([k for k in self.keywords if len(k) > 3]))


            return True

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red al acceder a {url}: {e}")
            print(f"Error de red al acceder a {url}: {e}")
            return False
        except Exception as e:
            st.error(f"Error inesperado al extraer metadatos de {url}: {e}")
            print(f"Error inesperado al extraer metadatos de {url}: {e}")
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
            "Si busco información sobre {keyword}, ¿qué sitios web debería visitar?",
            "Dame una lista de sitios web relevantes para {keyword}",
            "¿Qué recursos online existen sobre {keyword}?",
        ]

        # Usar las keywords extraídas/proporcionadas y la categoría si existe
        base_keywords = self.keywords.copy()
        if self.category:
            base_keywords.append(self.category)
            # También generar prompts específicos de la categoría si keywords están vacías
            if not self.keywords:
                 templates_cat = [
                    "¿Qué empresas son líderes en {keyword}?",
                    "¿Dónde encuentro información sobre {keyword}?",
                    "Recomiéndame sitios web sobre {keyword}"
                 ]
                 for t in templates_cat:
                     self.prompts.append(t.format(keyword=self.category))


        # Asegurar que tenemos keywords para generar prompts
        if not base_keywords:
             st.warning("No se pudieron determinar palabras clave. Usando el dominio como keyword.")
             # Como último recurso, usar el nombre del dominio (sin TLD) como keyword
             domain_name = urlparse(f"https://{self.domain}").netloc
             domain_parts = domain_name.split('.')
             keyword_from_domain = domain_parts[-2] if len(domain_parts) > 1 else domain_parts[0]
             base_keywords.append(keyword_from_domain)


        # Eliminar duplicados y asegurar que no sean demasiado genéricas
        valid_keywords = list(set([k for k in base_keywords if len(k) > 3 and k.lower() not in ['http', 'https', 'www']]))


        # Generar prompts hasta el límite deseado
        generated_count = 0
        for keyword in valid_keywords:
            if generated_count >= num_prompts:
                break
            template = templates[len(self.prompts) % len(templates)] # Ciclar por las plantillas
            self.prompts.append(template.format(keyword=keyword))
            generated_count += 1

        # Si no se generaron suficientes prompts, añadir algunos más genéricos si hay categoría
        if len(self.prompts) < num_prompts and self.category:
            generic_templates = [
                "Háblame sobre el sector de {keyword}",
                "¿Qué tendencias hay en {keyword}?"
            ]
            remaining = num_prompts - len(self.prompts)
            for i in range(min(remaining, len(generic_templates))):
                 self.prompts.append(generic_templates[i].format(keyword=self.category))

        # Asegurar que haya al menos un prompt
        if not self.prompts:
            st.error("No se pudieron generar prompts. Revisa el dominio o proporciona keywords.")
            # Añadir un prompt muy genérico como fallback si es necesario
            self.prompts.append(f"Háblame del sitio web {self.domain}")


        print(f"Generated prompts: {self.prompts}") # Para depuración
        return self.prompts

    def query_llm(self, provider, prompt):
        """Consultar a diferentes LLMs"""
        result = {"prompt": prompt, "response": None, "contains_domain": False, "position": -1, "has_link": False, "description": "", "error": None}

        # Verificar si la clave API para este proveedor está disponible
        api_key = None
        if provider == "openai" and openai_api_key:
            api_key = openai_api_key
        elif provider == "anthropic" and ANTHROPIC_API_KEY:
            api_key = ANTHROPIC_API_KEY
        elif provider == "gemini" and GEMINI_API_KEY:
            api_key = GEMINI_API_KEY

        if not api_key:
            result["error"] = f"API key for {provider} not configured."
            print(f"Skipping {provider} query due to missing API key.")
            # Devuelve el resultado con el error, pero no intentes la llamada API
            self.results[provider].append(result) # Asegúrate de añadirlo a los resultados para contarlo
            return result

        try:
            if provider == "openai":
                # Ya hemos asignado openai.api_key globalmente
                response = openai.chat.completions.create( # Uso de la nueva API v1+
                    model="gpt-3.5-turbo", # Usar un modelo más económico/rápido para pruebas? O gpt-4 si es necesario
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil que responde preguntas con información precisa y menciona sitios web relevantes cuando es apropiado."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5, # Un poco de variabilidad
                    max_tokens=500
                )
                result["response"] = response.choices[0].message.content

            elif provider == "anthropic":
                # Usa la clave específica para este proveedor
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01", # Requerido por la API
                    "content-type": "application/json"
                }
                # Estructura de mensaje recomendada para Claude 3 / Claude 2.1
                data = {
                     "model": "claude-3-haiku-20240307", # Modelo rápido y capaz
                     #"model": "claude-2.1", # Modelo anterior si prefieres
                     "max_tokens": 1000,
                     "messages": [
                        {"role": "user", "content": prompt}
                    ]
                 }
                api_url = "https://api.anthropic.com/v1/messages" # Endpoint actualizado
                response = requests.post(api_url, headers=headers, json=data, timeout=30)
                response.raise_for_status() # Lanza error si la respuesta no es 2xx
                response_data = response.json()
                # Extraer el texto de la respuesta
                if response_data.get("content") and isinstance(response_data["content"], list) and response_data["content"][0].get("type") == "text":
                     result["response"] = response_data["content"][0]["text"]
                else:
                     result["response"] = json.dumps(response_data) # Devolver JSON si la estructura no es la esperada
                     result["error"] = "Unexpected response structure from Anthropic."


            elif provider == "gemini":
                # Usa la clave específica para este proveedor
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}" # Modelo más reciente/rápido
                #url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}" # Modelo anterior
                data = {
                    "contents": [{"parts":[{"text": prompt}]}],
                     # Configuraciones de seguridad (opcional, ajustar según necesidad)
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                    ],
                    "generationConfig": {
                        "temperature": 0.6,
                        "maxOutputTokens": 500
                    }
                }
                headers = {'Content-Type': 'application/json'}
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                response_data = response.json()

                # Manejo robusto de la respuesta de Gemini
                if response_data.get("candidates") and response_data["candidates"][0].get("content", {}).get("parts"):
                    result["response"] = response_data["candidates"][0]["content"]["parts"][0].get("text", "")
                elif response_data.get("promptFeedback", {}).get("blockReason"):
                    # Si el contenido fue bloqueado por seguridad
                    reason = response_data["promptFeedback"]["blockReason"]
                    details = response_data["promptFeedback"].get("safetyRatings", [])
                    result["response"] = f"Blocked by Gemini Safety Filter: {reason}. Details: {details}"
                    result["error"] = f"Gemini safety block: {reason}"
                else:
                     # Si la estructura no es la esperada pero no hay error de bloqueo
                    result["response"] = json.dumps(response_data) # Devolver JSON como fallback
                    result["error"] = "Unexpected response structure from Gemini."


            # Analizar la respuesta si no hubo error en la llamada API
            if result["response"] and not result["error"]:
                self._analyze_response(result)

        except openai.APIError as e:
            # Manejar errores específicos de OpenAI
             st.error(f"Error de API OpenAI: {e}")
             print(f"Error de API OpenAI: {e}")
             result["error"] = f"OpenAI API Error: {e}"
        except requests.exceptions.RequestException as e:
            # Manejar errores de red/HTTP para Anthropic/Gemini
            st.error(f"Error de red consultando {provider}: {e}")
            print(f"Error de red consultando {provider}: {e}")
            result["error"] = f"Network Error ({provider}): {e}"
            # Intentar extraer mensaje de error de la respuesta si existe
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    result["error"] += f" - Details: {error_details}"
                except json.JSONDecodeError:
                     result["error"] += f" - Status Code: {e.response.status_code}, Response: {e.response.text[:200]}..." # Mostrar parte del texto si no es JSON

        except Exception as e:
            # Capturar cualquier otro error inesperado
            st.error(f"Error inesperado al consultar {provider}: {e}")
            print(f"Error inesperado al consultar {provider}: {e}")
            result["error"] = f"Unexpected Error ({provider}): {e}"

        # Asegurar que el resultado se añade incluso si hubo un error
        # (ya se hace dentro del bloque try/except y en el chequeo de API key)
        # self.results[provider].append(result) # Duplicado, eliminar
        return result


    def _analyze_response(self, result):
        """Analizar si la respuesta contiene el dominio y cómo aparece"""
        if not result or not result.get("response"):
            return # No hay respuesta para analizar

        response_text = result["response"]
        # Normalizar el dominio a 'example.com' para la búsqueda
        parsed_uri = urlparse(f"https://{self.domain}" if not self.domain.startswith('http') else self.domain)
        domain_name = parsed_uri.netloc.replace('www.', '') # Quitar www. para una coincidencia más amplia

        if not domain_name: # Caso borde: si el dominio es inválido
            return

        # Convertir respuesta a minúsculas para búsqueda insensible a mayúsculas
        response_lower = response_text.lower()
        domain_lower = domain_name.lower()

        # Verificar si contiene el nombre de dominio (sin TLD a veces es mejor)
        # domain_base = domain_lower.split('.')[0] # Ej: 'example' de 'example.com'
        # if domain_base in response_lower: ... (alternativa)

        # Verificar si contiene el dominio completo (ej: example.com)
        if domain_lower in response_lower:
            result["contains_domain"] = True

            # Encontrar la primera posición
            try:
                position = response_lower.find(domain_lower)
                if position != -1: # Asegurarse de que se encontró
                    result["position"] = position / len(response_text) if len(response_text) > 0 else 0
            except Exception:
                result["position"] = -1 # Error al calcular posición

            # Verificar si contiene un enlace completo (http/https)
            # Usar regex más específico para evitar coincidencias parciales (ej: sub.example.com)
            link_pattern = rf'https?://(?:[a-zA-Z0-9-]+\.)*{re.escape(domain_lower)}'
            result["has_link"] = bool(re.search(link_pattern, response_lower))

            # Extraer descripción contextual (100 caracteres antes y después)
            context_window = 100
            start = max(0, position - context_window)
            end = min(len(response_text), position + len(domain_name) + context_window)
            context_snippet = response_text[start:end]

            # Resaltar la mención del dominio en el snippet
            try:
                # Usar re.sub para reemplazo insensible a mayúsculas/minúsculas
                highlighted_snippet = re.sub(f"({re.escape(domain_name)})", r"**\1**", context_snippet, flags=re.IGNORECASE)
                result["description"] = f"...{highlighted_snippet}..."
            except Exception:
                 result["description"] = f"...{context_snippet}..." # Fallback sin resaltado

    def run_analysis(self, progress_bar):
        """Ejecutar el análisis completo con barra de progreso"""
        # 1. Extraer metadatos
        st.info("Extrayendo metadatos del sitio...")
        if not self.extract_metadata():
             st.error(f"No se pudieron extraer metadatos para {self.domain}. El análisis no puede continuar.")
             # Detener si la extracción falla y no hay keywords/categoría
             if not self.keywords and not self.category:
                 return {"error": "No se pudieron extraer los metadatos y no se proporcionaron keywords/categoría."}
             else:
                 st.warning("Continuando análisis con keywords/categoría proporcionadas.")
        else:
            st.success(f"Metadatos extraídos para {self.domain}")
            # Mostrar metadatos básicos extraídos
            with st.expander("Metadatos extraídos"):
                st.write(f"**Título:** {self.metadata.get('title', 'N/A')}")
                st.write(f"**Descripción:** {self.metadata.get('description', 'N/A')}")
                st.write(f"**Keywords inferidas:** {', '.join(self.keywords) if self.keywords else 'N/A'}")


        # 2. Generar prompts
        st.info("Generando prompts para los LLMs...")
        self.generate_prompts(num_prompts=5) # Generar hasta 5 prompts
        if not self.prompts:
             st.error("No se pudieron generar prompts. Verifica la configuración.")
             return {"error": "No se generaron prompts para el análisis."}
        st.success(f"Generados {len(self.prompts)} prompts.")
        with st.expander("Prompts generados"):
             st.write(self.prompts)


        # 3. Consultar LLMs
        st.info("Consultando a los modelos de lenguaje (OpenAI, Anthropic, Gemini)...")
        providers = []
        if openai_api_key: providers.append("openai")
        if ANTHROPIC_API_KEY: providers.append("anthropic")
        if GEMINI_API_KEY: providers.append("gemini")

        if not providers:
            st.error("No hay claves API configuradas para ningún LLM. No se pueden realizar consultas.")
            return {"error": "No hay LLMs configurados."}

        total_queries = len(self.prompts) * len(providers)
        query_count = 0

        # Reiniciar resultados
        self.results = {p: [] for p in providers}

        for i, prompt in enumerate(self.prompts):
            st.write(f"--- Prompt {i+1}/{len(self.prompts)}: `{prompt}` ---")
            for provider in providers:
                st.write(f"Consultando a {provider.capitalize()}...")
                # Pasar el objeto de la herramienta para acceder a la key correcta
                result = self.query_llm(provider, prompt)
                # No necesitas añadirlo aquí si ya lo haces en query_llm
                # self.results[provider].append(result)
                query_count += 1
                progress_percentage = query_count / total_queries
                progress_bar.progress(progress_percentage, text=f"Progreso: {query_count}/{total_queries} ({progress_percentage*100:.0f}%)")

                # Mostrar resultado parcial (opcional)
                if result.get("error"):
                     st.error(f"Error con {provider.capitalize()}: {result['error']}")
                elif result.get("response"):
                     with st.expander(f"Respuesta de {provider.capitalize()}", expanded=False):
                         st.markdown(f"**Menciona dominio:** {'Sí' if result.get('contains_domain') else 'No'} | **Tiene enlace:** {'Sí' if result.get('has_link') else 'No'}")
                         if result.get('contains_domain'):
                             st.markdown(f"**Contexto:** {result.get('description', 'N/A')}")
                         # st.text_area("Respuesta completa", result["response"], height=150) # Descomentar para ver respuesta completa

                # Pausa pequeña para evitar límites de rate muy estrictos
                time.sleep(1.5) # Aumentar ligeramente la pausa

        # 4. Calcular métricas
        st.info("Calculando métricas de rendimiento...")
        self.calculate_metrics()
        st.success("Métricas calculadas.")

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
        total_queries = 0
        # Iterar solo sobre los proveedores que realmente se consultaron (tienen resultados)
        for provider, provider_results in self.results.items():
            # Asegurarse de que provider_results es una lista (incluso si está vacía)
            if isinstance(provider_results, list):
                all_results.extend(provider_results)
                total_queries += len(provider_results) # Contar consultas realizadas efectivamente
            else:
                print(f"Advertencia: self.results['{provider}'] no es una lista. Contenido: {provider_results}")


        if not all_results or total_queries == 0:
            print("No hay resultados válidos para calcular métricas.")
            # Dejar métricas en 0 o valores por defecto
            self.metrics = {k: 0 for k in self.metrics}
            self.metrics["prominence_score"] = 1 # Peor caso para prominencia
            self.metrics["dataset_score"] = 0 # Sin info de dataset
            return

        # Filtrar resultados que no tuvieron errores de API/red
        valid_results = [r for r in all_results if r and r.get("response") is not None and r.get("error") is None]

        if not valid_results:
             print("No hay resultados válidos (sin errores) para calcular métricas.")
             self.metrics = {k: 0 for k in self.metrics}
             self.metrics["prominence_score"] = 1
             self.metrics["dataset_score"] = 0
             return

        # Tasa de inclusión (sobre consultas válidas)
        mentions = [r for r in valid_results if r.get("contains_domain")]
        self.metrics["inclusion_rate"] = len(mentions) / len(valid_results) if valid_results else 0

        # Puntuación de prominencia (promedio de 1 - posición relativa, más alto es mejor)
        # Solo considerar menciones donde la posición se calculó correctamente (>= 0)
        valid_positions = [r.get("position") for r in mentions if r.get("position", -1) >= 0]
        if valid_positions:
            # Invertir la métrica: 1 - posición (0 es inicio, 1 es final) -> promedio más cercano a 1 es mejor
            prominence_values = [1 - p for p in valid_positions]
            self.metrics["prominence_score"] = sum(prominence_values) / len(prominence_values)
        else:
            self.metrics["prominence_score"] = 0 # Si no hay menciones con posición válida, la prominencia es 0


        # Puntuación de citas (proporción de menciones que tienen enlace)
        citations = [r for r in mentions if r.get("has_link")]
        self.metrics["citation_score"] = len(citations) / len(mentions) if mentions else 0

        # Puntuación de dataset (simulada) - Podría mejorarse en el futuro
        # Podría basarse en la frecuencia de keywords o autoridad del dominio (requiere más APIs)
        self.metrics["dataset_score"] = 0.65 + (self.metrics["inclusion_rate"] * 0.1) # Ejemplo de cálculo simulado simple


    def generate_report(self):
        """Generar informe con recomendaciones"""
        # Asegurarse de que las métricas se han calculado
        if not self.metrics or self.metrics.get("inclusion_rate", 0) == 0 and len(self.prompts) > 0 :
             # Volver a calcular si es necesario (o si falló antes)
             self.calculate_metrics()

        # Calcular total de queries y menciones
        total_queries_attempted = 0
        mentions_count = 0
        all_results_list = []
        for provider, results_list in self.results.items():
            if isinstance(results_list, list):
                 total_queries_attempted += len(results_list)
                 all_results_list.extend(results_list)

        mentions = [r for r in all_results_list if r and r.get("contains_domain")]
        mentions_count = len(mentions)


        report = {
            "domain": self.domain,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self.metrics,
            "mentions_count": mentions_count,
            "total_queries_attempted": total_queries_attempted, # Total de intentos de consulta
             "prompts_used": self.prompts,
             "llms_used": list(self.results.keys()),
            "recommendations": [],
             "raw_results": self.results # Incluir resultados detallados
        }

        # --- Generar recomendaciones basadas en métricas ---

        # Recomendaciones sobre Tasa de Inclusión
        if self.metrics["inclusion_rate"] == 0:
             report["recommendations"].append({
                 "type": "critical",
                 "title": "Ausencia Total en Respuestas",
                 "description": f"Tu dominio '{self.domain}' no apareció en ninguna de las {total_queries_attempted} respuestas analizadas. Es crucial aumentar la visibilidad y autoridad de tu sitio en los temas relacionados ({', '.join(self.keywords[:3])}...). Considera crear contenido fundamental, obtener enlaces de calidad y asegurar que tu sitio sea técnicamente accesible."
             })
        elif self.metrics["inclusion_rate"] < 0.15:
            report["recommendations"].append({
                "type": "critical",
                "title": "Muy Baja Tasa de Inclusión",
                "description": f"Tu dominio solo aparece en {self.metrics['inclusion_rate']:.1%} de las respuestas. Esto sugiere una baja asociación por parte de los LLMs. Enfócate en crear contenido exhaustivo y de alta calidad sobre tus keywords principales. Promociona tu contenido para ganar autoridad."
            })
        elif self.metrics["inclusion_rate"] < 0.4:
            report["recommendations"].append({
                "type": "warning",
                "title": "Tasa de Inclusión Moderada",
                "description": f"Tu dominio aparece en {self.metrics['inclusion_rate']:.1%} de las respuestas. Hay margen de mejora. Revisa qué tipo de contenido mencionan los LLMs y crea más contenido similar o complementario. Optimiza el contenido existente para tus keywords."
            })
        else:
            report["recommendations"].append({
                "type": "info",
                "title": "Buena Tasa de Inclusión",
                "description": f"Tu dominio tiene una presencia sólida ({self.metrics['inclusion_rate']:.1%}) en las respuestas. ¡Buen trabajo! Mantén la calidad y frecuencia de tu contenido. Explora prompts más específicos para identificar oportunidades de nicho."
            })

        # Recomendaciones sobre Prominencia (Recordar: más alto es mejor ahora)
        if self.metrics["inclusion_rate"] > 0: # Solo dar consejos de prominencia si hay menciones
             if self.metrics["prominence_score"] < 0.3:
                 report["recommendations"].append({
                     "type": "warning",
                     "title": "Baja Prominencia en Respuestas",
                     "description": f"Cuando tu dominio es mencionado, tiende a aparecer tarde en la respuesta (prominencia: {self.metrics['prominence_score']:.2f}). Esto puede indicar que se considera relevante, pero no una fuente principal. Refuerza la autoridad en tus temas clave y asegúrate de que el contenido más importante sea fácilmente identificable."
                 })
             elif self.metrics["prominence_score"] < 0.6:
                 report["recommendations"].append({
                     "type": "improvement",
                     "title": "Prominencia Mejorable",
                     "description": f"Tu dominio aparece en posiciones intermedias (prominencia: {self.metrics['prominence_score']:.2f}). Intenta que tu marca/dominio se asocie más directamente con las respuestas a preguntas clave. Considera usar datos estructurados y optimizar títulos/encabezados."
                 })
             else:
                  report["recommendations"].append({
                     "type": "info",
                     "title": "Buena Prominencia",
                     "description": f"Tu dominio suele aparecer pronto en las respuestas relevantes (prominencia: {self.metrics['prominence_score']:.2f}), indicando que los LLMs lo consideran una fuente importante."
                 })


        # Recomendaciones sobre Citas con Enlaces
        if self.metrics["inclusion_rate"] > 0: # Solo dar consejos de citas si hay menciones
            if self.metrics["citation_score"] == 0:
                 report["recommendations"].append({
                     "type": "warning",
                     "title": "Sin Citas con Enlaces",
                     "description": "Aunque tu dominio es mencionado, ninguna de esas menciones incluyó un enlace directo. Esto limita el tráfico de referencia. Asegúrate de que tu nombre de dominio sea único y fácil de asociar con tu sitio. Podría ser un indicio de que los LLMs citan otras fuentes al hablar de temas donde deberías ser referente."
                 })
            elif self.metrics["citation_score"] < 0.3:
                report["recommendations"].append({
                    "type": "improvement",
                    "title": "Pocas Citas con Enlaces",
                    "description": f"Solo {self.metrics['citation_score']:.1%} de las menciones incluyen un enlace. Revisa la calidad y unicidad del contenido que podría ser enlazado. Fomenta enlaces naturales desde otros sitios de autoridad, ya que esto puede influir en cómo los LLMs citan."
                })
            else:
                 report["recommendations"].append({
                     "type": "info",
                     "title": "Buenas Citas con Enlaces",
                     "description": f"Una buena proporción ({self.metrics['citation_score']:.1%}) de las menciones incluyen enlaces, lo cual es positivo para la visibilidad y el posible tráfico."
                 })

        # Recomendación general basada en Dataset Score (Simulado)
        if self.metrics["dataset_score"] < 0.6:
             report["recommendations"].append({
                 "type": "info",
                 "title": "Potencial en Datasets de Entrenamiento",
                 "description": "La puntuación simulada de presencia en datasets es moderada. Aumentar la autoridad general de tu dominio y la cantidad de contenido único y valioso podría mejorar tu inclusión en futuros entrenamientos de LLMs."
             })

        return report


# --- Interfaz de Streamlit (Modificada para usar barra de progreso y mostrar resultados) ---
def create_streamlit_app():
    st.set_page_config(layout="wide") # Usar más espacio
    st.title("🤖 AI Linkability Analyzer v1.1")
    st.write("Analiza cómo los LLMs (GPT, Claude, Gemini) mencionan tu sitio web.")
    st.caption("Introduce un dominio y, opcionalmente, palabras clave o categoría para generar prompts y consultar a las IAs.")

    # Comprobar si hay claves API configuradas al inicio
    configured_llms = []
    if openai_api_key: configured_llms.append("OpenAI")
    if ANTHROPIC_API_KEY: configured_llms.append("Anthropic")
    if GEMINI_API_KEY: configured_llms.append("Gemini")

    if not configured_llms:
        st.error("❌ **Error de Configuración:** No se ha detectado ninguna clave API en los secretos (local: `.streamlit/secrets.toml`, Cloud: configuración de la app). Añade las claves para poder realizar el análisis.")
        st.info("""
        **Ejemplo para `.streamlit/secrets.toml`:**
        ```toml
        [api_keys]
        openai = "sk-..."
        anthropic = "sk-ant-..."
        gemini = "AIza..."
        ```
        **En Streamlit Cloud:** Añade los secretos con los nombres `api_keys.openai`, `api_keys.anthropic`, `api_keys.gemini`.
        """)
        st.stop() # Detener la app si no hay claves
    else:
        st.success(f"✅ Claves API detectadas para: {', '.join(configured_llms)}")


    with st.form("analysis_form"):
        domain = st.text_input("Dominio web (sin http/https)", placeholder="ejemplo.com")
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_area("Palabras clave (opcional, una por línea)", placeholder="marketing digital\nseo local\n...")
        with col2:
            category = st.text_input("Categoría del negocio (opcional)", placeholder="agencia de marketing digital")

        submitted = st.form_submit_button(f"🚀 Analizar Dominio con {', '.join(configured_llms)}")

        if submitted:
            if not domain:
                st.error("Por favor, introduce un dominio web.")
            else:
                # Limpiar dominio por si acaso
                domain = domain.replace("https://", "").replace("http://", "").split('/')[0]

                keywords_list = [k.strip() for k in keywords.split('\n') if k.strip()]

                st.info(f"Iniciando análisis para: **{domain}**")
                progress_bar = st.progress(0, text="Iniciando...")

                try:
                    tool = AILinkabilityTool(domain, keywords_list, category)
                    # Pasar la barra de progreso a run_analysis
                    analysis_output = tool.run_analysis(progress_bar)

                    # Verificar si hubo un error durante el análisis
                    if analysis_output and "error" in analysis_output:
                         st.error(f"El análisis falló: {analysis_output['error']}")
                         st.stop() # Detener si el análisis falló críticamente


                    st.success("📊 ¡Análisis completado!")
                    progress_bar.progress(1.0, text="Completado")

                    # Generar y mostrar el informe
                    report = tool.generate_report()

                    # --- Mostrar Resultados ---
                    st.header("📈 Resumen de Métricas")
                    metrics = report['metrics']
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Tasa Inclusión", f"{metrics['inclusion_rate']:.1%}",
                                  help="Porcentaje de respuestas de LLMs que mencionaron tu dominio.")
                    m_col2.metric("Prominencia", f"{metrics['prominence_score']:.2f}",
                                  help="Qué tan temprano aparece tu dominio en la respuesta (1.0 = al inicio, 0.0 = al final/no aparece). Más alto es mejor.")
                    m_col3.metric("Citas (con Link)", f"{metrics['citation_score']:.1%}",
                                  help="Porcentaje de menciones que incluyeron un enlace directo a tu sitio.")
                    m_col4.metric("Score Dataset (Sim.)", f"{metrics['dataset_score']:.1%}",
                                  help="Puntuación simulada de presencia en datasets de entrenamiento (valor indicativo).")


                    st.header("💡 Recomendaciones")
                    if report["recommendations"]:
                        for rec in report["recommendations"]:
                            rec_type = rec.get("type", "info")
                            rec_title = rec.get("title", "Recomendación")
                            rec_desc = rec.get("description", "")
                            if rec_type == "critical":
                                st.error(f"**{rec_title}**: {rec_desc}")
                            elif rec_type == "warning":
                                st.warning(f"**{rec_title}**: {rec_desc}")
                            else: # improvement o info
                                st.info(f"**{rec_title}**: {rec_desc}")
                    else:
                        st.info("No se generaron recomendaciones específicas basadas en los resultados.")


                    st.header(f"💬 Menciones Detectadas ({report['mentions_count']} / {report['total_queries_attempted']} consultas)")
                    mentions_data = []
                    for provider, results_list in tool.results.items():
                         # Asegurarse de que es una lista
                         if isinstance(results_list, list):
                             for result in results_list:
                                 if result and result.get("contains_domain"):
                                     mentions_data.append({
                                         "LLM": provider.capitalize(),
                                         "Prompt": result.get("prompt", "N/A"),
                                         "Tiene Enlace": "✔️ Sí" if result.get("has_link") else "❌ No",
                                         "Contexto de la Mención": result.get("description", "N/A"),
                                         #"Respuesta Completa": result.get("response", "N/A") # Opcional: añadir respuesta completa
                                     })

                    if mentions_data:
                        # Usar st.dataframe para tabla interactiva
                        df_mentions = pd.DataFrame(mentions_data)
                        st.dataframe(df_mentions, use_container_width=True)

                        # Opción para ver detalles de una mención específica
                        # selected_mention = st.selectbox("Selecciona una mención para ver detalles", options=df_mentions.index, format_func=lambda x: f"Mención {x+1} ({df_mentions.loc[x, 'LLM']})")
                        # if selected_mention is not None:
                        #      st.markdown(f"**Prompt:** `{df_mentions.loc[selected_mention, 'Prompt']}`")
                        #      st.markdown(f"**Contexto:** {df_mentions.loc[selected_mention, 'Contexto de la Mención']}")
                        #      # st.text_area("Respuesta Completa", df_mentions.loc[selected_mention, "Respuesta Completa"], height=200)

                    else:
                        st.info("No se encontraron menciones explícitas de tu dominio en las respuestas analizadas.")


                    # --- Sección para descargar el informe ---
                    st.header("📥 Descargar Informe")
                    report_json = json.dumps(report, indent=2, ensure_ascii=False)
                    # Limpiar nombre de archivo
                    safe_domain = re.sub(r'[^a-zA-Z0-9\-]+', '_', domain)
                    file_name = f"linkability_report_{safe_domain}_{time.strftime('%Y%m%d_%H%M')}.json"

                    st.download_button(
                        label="Descargar Informe Completo (JSON)",
                        data=report_json,
                        file_name=file_name,
                        mime="application/json"
                    )
                    with st.expander("Ver contenido del JSON del informe"):
                        st.json(report)


                except Exception as e:
                    st.error(f"Ocurrió un error general durante el análisis: {e}")
                    import traceback
                    st.exception(e) # Muestra el traceback completo para depuración
                    progress_bar.progress(1.0, text="Error")


if __name__ == "__main__":
    create_streamlit_app()