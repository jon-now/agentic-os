import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ResearchDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP_DIVE = "deep_dive"

class ResearchPipeline:
    """Advanced research pipeline with multi-stage processing and content creation"""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.research_cache = {}
        self.active_research_sessions = {}
        self.research_templates = self._load_research_templates()

    def _load_research_templates(self) -> Dict:
        """Load research workflow templates"""
        return {
            "academic_research": {
                "sources": ["google_scholar", "arxiv", "pubmed", "wikipedia"],
                "depth": ResearchDepth.COMPREHENSIVE,
                "output_format": "academic_report",
                "citation_style": "apa"
            },
            "market_research": {
                "sources": ["google_search", "industry_reports", "news", "company_websites"],
                "depth": ResearchDepth.STANDARD,
                "output_format": "market_analysis",
                "focus_areas": ["trends", "competitors", "opportunities"]
            },
            "technical_research": {
                "sources": ["github", "documentation", "stack_overflow", "technical_blogs"],
                "depth": ResearchDepth.DEEP_DIVE,
                "output_format": "technical_guide",
                "include_code_examples": True
            },
            "general_research": {
                "sources": ["wikipedia", "google_search", "news"],
                "depth": ResearchDepth.STANDARD,
                "output_format": "summary_report",
                "accessibility_level": "general_audience"
            }
        }

    async def conduct_research(self, topic: str, research_type: str = "general_research",
                             custom_parameters: Optional[Dict] = None) -> Dict:
        """Conduct comprehensive research on a topic"""
        try:
            research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get research template
            template = self.research_templates.get(research_type, self.research_templates["general_research"])

            # Apply custom parameters
            if custom_parameters:
                template = {**template, **custom_parameters}

            # Initialize research session
            research_session = {
                "id": research_id,
                "topic": topic,
                "type": research_type,
                "template": template,
                "started_at": datetime.now().isoformat(),
                "status": "in_progress",
                "stages_completed": [],
                "results": {}
            }

            self.active_research_sessions[research_id] = research_session

            # Execute research pipeline
            results = await self._execute_research_pipeline(research_session)

            # Update session with results
            research_session["results"] = results
            research_session["status"] = "completed"
            research_session["completed_at"] = datetime.now().isoformat()

            # Cache results
            self.research_cache[research_id] = research_session

            # Remove from active sessions
            if research_id in self.active_research_sessions:
                del self.active_research_sessions[research_id]

            return {
                "research_id": research_id,
                "topic": topic,
                "type": research_type,
                "results": results,
                "status": "completed",
                "duration": self._calculate_duration(research_session)
            }

        except Exception as e:
            logger.error("Research pipeline failed: %s", e)
            return {"error": str(e)}

    async def _execute_research_pipeline(self, research_session: Dict) -> Dict:
        """Execute the multi-stage research pipeline"""
        template = research_session["template"]
        topic = research_session["topic"]

        pipeline_results = {
            "raw_data": {},
            "processed_data": {},
            "synthesis": {},
            "content_outputs": {},
            "metadata": {}
        }

        # Stage 1: Data Collection
        logger.info("Stage 1: Collecting data for '%s'", topic)
        raw_data = await self._collect_research_data(topic, template)
        pipeline_results["raw_data"] = raw_data
        research_session["stages_completed"].append("data_collection")

        # Stage 2: Data Processing and Analysis
        logger.info("Stage 2: Processing and analyzing data")
        processed_data = await self._process_research_data(raw_data, template)
        pipeline_results["processed_data"] = processed_data
        research_session["stages_completed"].append("data_processing")

        # Stage 3: Synthesis and Insights
        logger.info("Stage 3: Synthesizing insights")
        synthesis = await self._synthesize_research_insights(processed_data, topic, template)
        pipeline_results["synthesis"] = synthesis
        research_session["stages_completed"].append("synthesis")

        # Stage 4: Content Generation
        logger.info("Stage 4: Generating content outputs")
        content_outputs = await self._generate_content_outputs(synthesis, template)
        pipeline_results["content_outputs"] = content_outputs
        research_session["stages_completed"].append("content_generation")

        # Stage 5: Quality Assessment
        logger.info("Stage 5: Assessing quality and completeness")
        quality_assessment = await self._assess_research_quality(pipeline_results)
        pipeline_results["metadata"]["quality_assessment"] = quality_assessment
        research_session["stages_completed"].append("quality_assessment")

        return pipeline_results

    async def _collect_research_data(self, topic: str, template: Dict) -> Dict:
        """Collect raw research data from multiple sources"""
        raw_data = {
            "sources": {},
            "collection_metadata": {
                "topic": topic,
                "sources_attempted": [],
                "sources_successful": [],
                "collection_time": datetime.now().isoformat()
            }
        }

        sources = template.get("sources", ["google_search", "wikipedia"])

        for source in sources:
            try:
                logger.info("Collecting data from %s", source)
                source_data = await self._collect_from_source(topic, source, template)

                if source_data and "error" not in source_data:
                    raw_data["sources"][source] = source_data
                    raw_data["collection_metadata"]["sources_successful"].append(source)
                else:
                    logger.warning("Failed to collect from {source}: %s", source_data.get('error', 'Unknown error'))

                raw_data["collection_metadata"]["sources_attempted"].append(source)

                # Add delay between sources to be respectful
                await asyncio.sleep(2)

            except Exception as e:
                logger.error("Error collecting from {source}: %s", e)
                raw_data["collection_metadata"]["sources_attempted"].append(source)

        return raw_data

    async def _collect_from_source(self, topic: str, source: str, template: Dict) -> Dict:
        """Collect data from a specific source"""
        if not self.orchestrator:
            return {"error": "No orchestrator available"}

        try:
            if source in ["google_search", "wikipedia", "google_scholar"]:
                # Use browser controller for web-based sources
                browser_controller = self.orchestrator.app_controllers.get("browser")
                if browser_controller:
                    return await browser_controller.research_topic(topic, template.get("depth", "standard"))

            elif source == "arxiv":
                return await self._collect_from_arxiv(topic)

            elif source == "github":
                return await self._collect_from_github(topic)

            elif source == "news":
                return await self._collect_from_news(topic)

            elif source == "documentation":
                return await self._collect_from_documentation(topic)

            else:
                return {"error": f"Unknown source: {source}"}

        except Exception as e:
            logger.error("Source collection failed for {source}: %s", e)
            return {"error": str(e)}

    async def _process_research_data(self, raw_data: Dict, template: Dict) -> Dict:
        """Process and structure raw research data"""
        processed_data = {
            "structured_content": {},
            "key_themes": [],
            "credibility_scores": {},
            "content_gaps": [],
            "processing_metadata": {
                "processed_at": datetime.now().isoformat(),
                "processing_method": "hybrid_analysis"
            }
        }

        sources_data = raw_data.get("sources", {})

        # Process each source
        for source_name, source_data in sources_data.items():
            try:
                # Structure content from this source
                structured = await self._structure_source_content(source_data, source_name)
                processed_data["structured_content"][source_name] = structured

                # Calculate credibility score
                credibility = self._calculate_source_credibility(source_data, source_name)
                processed_data["credibility_scores"][source_name] = credibility

            except Exception as e:
                logger.error("Error processing {source_name}: %s", e)

        # Extract key themes across all sources
        processed_data["key_themes"] = await self._extract_key_themes(processed_data["structured_content"])

        # Identify content gaps
        processed_data["content_gaps"] = await self._identify_content_gaps(
            processed_data["structured_content"], template
        )

        return processed_data

    async def _synthesize_research_insights(self, processed_data: Dict, topic: str, template: Dict) -> Dict:
        """Synthesize insights from processed research data"""
        synthesis = {
            "executive_summary": "",
            "key_findings": [],
            "insights": [],
            "recommendations": [],
            "confidence_assessment": {},
            "synthesis_metadata": {
                "synthesized_at": datetime.now().isoformat(),
                "synthesis_method": "llm_enhanced"
            }
        }

        try:
            # Generate executive summary
            synthesis["executive_summary"] = await self._generate_executive_summary(
                processed_data, topic, template
            )

            # Extract key findings
            synthesis["key_findings"] = await self._extract_key_findings(processed_data)

            # Generate insights
            synthesis["insights"] = await self._generate_insights(processed_data, template)

            # Generate recommendations
            synthesis["recommendations"] = await self._generate_recommendations(
                processed_data, template
            )

            # Assess confidence in findings
            synthesis["confidence_assessment"] = await self._assess_synthesis_confidence(
                processed_data, synthesis
            )

        except Exception as e:
            logger.error("Research synthesis failed: %s", e)
            synthesis["error"] = str(e)

        return synthesis

    async def _generate_content_outputs(self, synthesis: Dict, template: Dict) -> Dict:
        """Generate various content outputs from research synthesis"""
        content_outputs = {
            "formats": {},
            "generation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "output_format": template.get("output_format", "summary_report")
            }
        }

        output_format = template.get("output_format", "summary_report")

        try:
            # Generate primary output format
            if output_format == "academic_report":
                content_outputs["formats"]["academic_report"] = await self._generate_academic_report(synthesis, template)
            elif output_format == "market_analysis":
                content_outputs["formats"]["market_analysis"] = await self._generate_market_analysis(synthesis, template)
            elif output_format == "technical_guide":
                content_outputs["formats"]["technical_guide"] = await self._generate_technical_guide(synthesis, template)
            else:
                content_outputs["formats"]["summary_report"] = await self._generate_summary_report(synthesis, template)

            # Generate additional formats
            content_outputs["formats"]["bullet_summary"] = await self._generate_bullet_summary(synthesis)
            content_outputs["formats"]["presentation_outline"] = await self._generate_presentation_outline(synthesis)
            content_outputs["formats"]["social_media_summary"] = await self._generate_social_media_summary(synthesis)

        except Exception as e:
            logger.error("Content generation failed: %s", e)
            content_outputs["error"] = str(e)

        return content_outputs

    async def _assess_research_quality(self, pipeline_results: Dict) -> Dict:
        """Assess the quality and completeness of research"""
        quality_assessment = {
            "completeness_score": 0.0,
            "credibility_score": 0.0,
            "depth_score": 0.0,
            "coverage_score": 0.0,
            "overall_quality": 0.0,
            "quality_indicators": [],
            "improvement_suggestions": []
        }

        try:
            # Assess completeness
            raw_data = pipeline_results.get("raw_data", {})
            sources_successful = len(raw_data.get("sources", {}))
            sources_attempted = len(raw_data.get("collection_metadata", {}).get("sources_attempted", []))

            if sources_attempted > 0:
                quality_assessment["completeness_score"] = sources_successful / sources_attempted

            # Assess credibility
            processed_data = pipeline_results.get("processed_data", {})
            credibility_scores = processed_data.get("credibility_scores", {})

            if credibility_scores:
                avg_credibility = sum(credibility_scores.values()) / len(credibility_scores)
                quality_assessment["credibility_score"] = avg_credibility

            # Assess depth
            synthesis = pipeline_results.get("synthesis", {})
            key_findings = synthesis.get("key_findings", [])
            insights = synthesis.get("insights", [])

            depth_indicators = len(key_findings) + len(insights)
            quality_assessment["depth_score"] = min(depth_indicators / 10, 1.0)  # Normalize to 0-1

            # Assess coverage
            key_themes = processed_data.get("key_themes", [])
            quality_assessment["coverage_score"] = min(len(key_themes) / 5, 1.0)  # Normalize to 0-1

            # Calculate overall quality
            scores = [
                quality_assessment["completeness_score"],
                quality_assessment["credibility_score"],
                quality_assessment["depth_score"],
                quality_assessment["coverage_score"]
            ]
            quality_assessment["overall_quality"] = sum(scores) / len(scores)

            # Generate quality indicators
            if quality_assessment["overall_quality"] > 0.8:
                quality_assessment["quality_indicators"].append("high_quality_research")
            if quality_assessment["credibility_score"] > 0.7:
                quality_assessment["quality_indicators"].append("credible_sources")
            if quality_assessment["depth_score"] > 0.6:
                quality_assessment["quality_indicators"].append("comprehensive_analysis")

            # Generate improvement suggestions
            if quality_assessment["completeness_score"] < 0.7:
                quality_assessment["improvement_suggestions"].append("Consider additional data sources")
            if quality_assessment["depth_score"] < 0.5:
                quality_assessment["improvement_suggestions"].append("Expand analysis with more detailed insights")

        except Exception as e:
            logger.error("Quality assessment failed: %s", e)
            quality_assessment["error"] = str(e)

        return quality_assessment

    async def create_research_workflow(self, topic: str, deliverables: List[str],
                                     deadline: Optional[str] = None) -> Dict:
        """Create a complete research workflow with multiple deliverables"""
        try:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            workflow = {
                "id": workflow_id,
                "topic": topic,
                "deliverables": deliverables,
                "deadline": deadline,
                "created_at": datetime.now().isoformat(),
                "status": "planning",
                "stages": [],
                "outputs": {}
            }

            # Plan workflow stages
            workflow["stages"] = await self._plan_workflow_stages(topic, deliverables, deadline)

            # Execute workflow
            workflow["status"] = "executing"
            execution_results = await self._execute_research_workflow(workflow)

            workflow["outputs"] = execution_results
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now().isoformat()

            return workflow

        except Exception as e:
            logger.error("Research workflow creation failed: %s", e)
            return {"error": str(e)}

    async def _plan_workflow_stages(self, topic: str, deliverables: List[str],
                                  deadline: Optional[str] = None) -> List[Dict]:
        """Plan the stages of a research workflow"""
        stages = []

        # Stage 1: Initial Research
        stages.append({
            "stage": 1,
            "name": "initial_research",
            "description": f"Conduct initial research on {topic}",
            "estimated_duration": 30,  # minutes
            "dependencies": [],
            "outputs": ["research_data", "source_analysis"]
        })

        # Stage 2: Deep Dive Research (if comprehensive deliverables)
        if any(deliverable in ["report", "analysis", "guide"] for deliverable in deliverables):
            stages.append({
                "stage": 2,
                "name": "deep_dive_research",
                "description": "Conduct detailed research with specialized sources",
                "estimated_duration": 45,
                "dependencies": ["initial_research"],
                "outputs": ["detailed_data", "expert_insights"]
            })

        # Stage 3: Content Creation
        for deliverable in deliverables:
            stages.append({
                "stage": len(stages) + 1,
                "name": f"create_{deliverable}",
                "description": f"Create {deliverable} from research data",
                "estimated_duration": 20,
                "dependencies": ["initial_research"],
                "outputs": [deliverable]
            })

        # Stage 4: Review and Refinement
        stages.append({
            "stage": len(stages) + 1,
            "name": "review_and_refine",
            "description": "Review outputs and refine based on quality assessment",
            "estimated_duration": 15,
            "dependencies": [f"create_{d}" for d in deliverables],
            "outputs": ["final_outputs", "quality_report"]
        })

        return stages

    async def _execute_research_workflow(self, workflow: Dict) -> Dict:
        """Execute a complete research workflow"""
        outputs = {}
        topic = workflow["topic"]
        deliverables = workflow["deliverables"]

        try:
            # Execute initial research
            research_results = await self.conduct_research(topic, "general_research")
            outputs["research_data"] = research_results

            # Generate each deliverable
            for deliverable in deliverables:
                if deliverable == "document":
                    outputs["document"] = await self._create_research_document(research_results)
                elif deliverable == "presentation":
                    outputs["presentation"] = await self._create_research_presentation(research_results)
                elif deliverable == "summary":
                    outputs["summary"] = await self._create_research_summary(research_results)
                elif deliverable == "report":
                    outputs["report"] = await self._create_detailed_report(research_results)
                elif deliverable == "social_post":
                    outputs["social_post"] = await self._create_social_media_content(research_results)

            return outputs

        except Exception as e:
            logger.error("Workflow execution failed: %s", e)
            return {"error": str(e)}

    async def _create_research_document(self, research_results: Dict) -> Dict:
        """Create a formatted research document"""
        if not self.orchestrator:
            return {"error": "No orchestrator available"}

        try:
            document_controller = self.orchestrator.app_controllers.get("document")
            if not document_controller:
                return {"error": "Document controller not available"}

            # Extract research data
            results = research_results.get("results", {})
            synthesis = results.get("synthesis", {})

            # Create document
            doc_result = await document_controller.create_document("writer")
            if "error" in doc_result:
                return doc_result

            doc_id = doc_result["document_id"]

            # Add content sections
            topic = research_results.get("topic", "Research Topic")
            await document_controller.add_content(doc_id, f"Research Report: {topic}", "heading")

            # Executive summary
            executive_summary = synthesis.get("executive_summary", "")
            if executive_summary:
                await document_controller.add_content(doc_id, "Executive Summary", "heading")
                await document_controller.add_content(doc_id, executive_summary, "text")

            # Key findings
            key_findings = synthesis.get("key_findings", [])
            if key_findings:
                await document_controller.add_content(doc_id, "Key Findings", "heading")
                findings_text = "\n".join(key_findings)
                await document_controller.add_content(doc_id, findings_text, "list")

            # Insights and recommendations
            insights = synthesis.get("insights", [])
            if insights:
                await document_controller.add_content(doc_id, "Insights", "heading")
                insights_text = "\n".join(insights)
                await document_controller.add_content(doc_id, insights_text, "list")

            # Save document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{topic.replace(' ', '_')}_{timestamp}.txt"
            save_result = await document_controller.save_document(doc_id, filename)

            return {
                "document_id": doc_id,
                "filename": filename,
                "sections": ["executive_summary", "key_findings", "insights"],
                "status": "created",
                "save_result": save_result
            }

        except Exception as e:
            logger.error("Document creation failed: %s", e)
            return {"error": str(e)}

    async def _create_research_presentation(self, research_results: Dict) -> Dict:
        """Create a presentation from research results"""
        try:
            # Extract key data for presentation
            results = research_results.get("results", {})
            synthesis = results.get("synthesis", {})

            presentation_data = {
                "title": f"Research: {research_results.get('topic', 'Unknown Topic')}",
                "slides": []
            }

            # Title slide
            presentation_data["slides"].append({
                "slide_number": 1,
                "type": "title",
                "title": presentation_data["title"],
                "subtitle": f"Research conducted on {datetime.now().strftime('%B %d, %Y')}"
            })

            # Executive summary slide
            executive_summary = synthesis.get("executive_summary", "")
            if executive_summary:
                presentation_data["slides"].append({
                    "slide_number": 2,
                    "type": "content",
                    "title": "Executive Summary",
                    "content": executive_summary[:300] + "..." if len(executive_summary) > 300 else executive_summary
                })

            # Key findings slides
            key_findings = synthesis.get("key_findings", [])
            if key_findings:
                presentation_data["slides"].append({
                    "slide_number": 3,
                    "type": "bullet_points",
                    "title": "Key Findings",
                    "bullets": key_findings[:5]  # Limit to 5 key findings
                })

            # Insights slide
            insights = synthesis.get("insights", [])
            if insights:
                presentation_data["slides"].append({
                    "slide_number": 4,
                    "type": "bullet_points",
                    "title": "Key Insights",
                    "bullets": insights[:5]
                })

            # Recommendations slide
            recommendations = synthesis.get("recommendations", [])
            if recommendations:
                presentation_data["slides"].append({
                    "slide_number": 5,
                    "type": "bullet_points",
                    "title": "Recommendations",
                    "bullets": recommendations[:5]
                })

            return {
                "presentation_data": presentation_data,
                "slide_count": len(presentation_data["slides"]),
                "status": "created"
            }

        except Exception as e:
            logger.error("Presentation creation failed: %s", e)
            return {"error": str(e)}

    async def _create_research_summary(self, research_results: Dict) -> Dict:
        """Create a concise research summary"""
        try:
            results = research_results.get("results", {})
            synthesis = results.get("synthesis", {})

            summary = {
                "topic": research_results.get("topic", "Unknown Topic"),
                "summary_text": "",
                "key_points": [],
                "sources_count": 0,
                "confidence_level": "medium"
            }

            # Generate summary text
            executive_summary = synthesis.get("executive_summary", "")
            key_findings = synthesis.get("key_findings", [])

            summary_parts = []
            if executive_summary:
                summary_parts.append(executive_summary[:200] + "...")

            if key_findings:
                summary_parts.append("Key findings include: " + "; ".join(key_findings[:3]))

            summary["summary_text"] = " ".join(summary_parts)

            # Extract key points
            summary["key_points"] = key_findings[:5] + synthesis.get("insights", [])[:3]

            # Count sources
            raw_data = results.get("raw_data", {})
            summary["sources_count"] = len(raw_data.get("sources", {}))

            # Assess confidence
            confidence_assessment = synthesis.get("confidence_assessment", {})
            overall_confidence = confidence_assessment.get("overall_confidence", 0.5)

            if overall_confidence > 0.8:
                summary["confidence_level"] = "high"
            elif overall_confidence > 0.6:
                summary["confidence_level"] = "medium"
            else:
                summary["confidence_level"] = "low"

            return summary

        except Exception as e:
            logger.error("Summary creation failed: %s", e)
            return {"error": str(e)}

    async def _generate_executive_summary(self, processed_data: Dict, topic: str, template: Dict) -> str:
        """Generate executive summary using LLM"""
        try:
            if not self.orchestrator or not self.orchestrator.llm_client:
                return self._generate_fallback_summary(processed_data, topic)

            # Prepare data for LLM
            key_themes = processed_data.get("key_themes", [])
            structured_content = processed_data.get("structured_content", {})

            # Create prompt for executive summary
            prompt = """Generate a concise executive summary for research on "{topic}".

Key themes identified: {', '.join(key_themes[:5])}

Based on the research data, provide a 2-3 sentence executive summary that captures:
1. The main findings about {topic}
2. The most important insights
3. Key implications or significance

Keep it professional and accessible to a general audience."""

            summary = await self.orchestrator.llm_client.generate_response(prompt)
            return summary.strip()

        except Exception as e:
            logger.error("Executive summary generation failed: %s", e)
            return self._generate_fallback_summary(processed_data, topic)

    def _generate_fallback_summary(self, processed_data: Dict, topic: str) -> str:
        """Generate fallback summary without LLM"""
        key_themes = processed_data.get("key_themes", [])

        if key_themes:
            themes_text = ", ".join(key_themes[:3])
            return f"Research on {topic} reveals key themes including {themes_text}. The analysis provides insights into current trends and developments in this area."
        else:
            return f"Research conducted on {topic} with analysis of available sources and data."

    def _calculate_duration(self, research_session: Dict) -> float:
        """Calculate research session duration in minutes"""
        try:
            started = datetime.fromisoformat(research_session["started_at"])
            completed = datetime.fromisoformat(research_session.get("completed_at", datetime.now().isoformat()))
            return (completed - started).total_seconds() / 60
        except Exception:
            return 0.0

    def get_research_status(self, research_id: str) -> Optional[Dict]:
        """Get status of a research session"""
        if research_id in self.active_research_sessions:
            return self.active_research_sessions[research_id]
        elif research_id in self.research_cache:
            return self.research_cache[research_id]
        else:
            return None

    def list_research_sessions(self) -> Dict:
        """List all research sessions"""
        return {
            "active_sessions": list(self.active_research_sessions.keys()),
            "completed_sessions": list(self.research_cache.keys()),
            "total_sessions": len(self.active_research_sessions) + len(self.research_cache)
        }
